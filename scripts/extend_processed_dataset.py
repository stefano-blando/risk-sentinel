#!/usr/bin/env python3
"""Extend a RiskSentinel processed dataset by downloading only the missing delta."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import data_loader
from src.core.data_refresh import (
    build_network_features,
    build_regime_frame,
    compute_incremental_network_outputs,
    write_processed_dataset,
)

DELISTING_EVENTS = {
    "IPG": {
        "inactive_from": "2025-11-27",
        "reason": "Interpublic acquired by Omnicom; ticker inactive after merger close.",
        "successor_ticker": "OMC",
    },
    "K": {
        "inactive_from": "2025-12-12",
        "reason": "Kellanova acquired by Mars; ticker delisted after acquisition close.",
        "successor_ticker": None,
    },
}


def _extract_close_panel(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No rows returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = list(raw.columns.get_level_values(0))
        level1 = list(raw.columns.get_level_values(1))
        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("Unable to locate Close prices in yfinance response.")
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]}).copy()
    else:
        raise ValueError("Unsupported yfinance response format.")

    close = close.rename_axis(index="Date").sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.loc[:, [col for col in close.columns if col in symbols]]
    close = close.dropna(how="all")
    return close


def _download_close_prices(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(
        tickers=symbols,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    return _extract_close_panel(raw, symbols)


def _combine_history(base: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([base, delta]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.index = pd.to_datetime(combined.index).tz_localize(None)
    return combined


def _apply_delisting_policy(close_prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, str | None]]]:
    adjusted = close_prices.copy()
    inactive_summary: dict[str, dict[str, str | None]] = {}
    for ticker, meta in DELISTING_EVENTS.items():
        if ticker not in adjusted.columns:
            continue
        inactive_from = pd.Timestamp(meta["inactive_from"])
        adjusted.loc[adjusted.index >= inactive_from, ticker] = pd.NA
        last_valid = adjusted[ticker].last_valid_index()
        effective_inactive_from = None
        if last_valid is not None:
            later = adjusted.index[adjusted.index > last_valid]
            if len(later) > 0:
                effective_inactive_from = str(pd.Timestamp(later[0]).date())
        inactive_summary[ticker] = {
            "inactive_from": str(inactive_from.date()),
            "last_observed_date": str(pd.Timestamp(last_valid).date()) if last_valid is not None else None,
            "effective_inactive_from": effective_inactive_from,
            "reason": str(meta["reason"]),
            "successor_ticker": meta["successor_ticker"],
        }
    return adjusted, inactive_summary


def extend_processed_dataset(base_dir: Path, output_dir: Path, end_date: str, threshold: float, window: int) -> dict[str, object]:
    sector_mapping = pd.read_parquet(base_dir / "sector_mapping.parquet")
    close_prices = pd.read_parquet(base_dir / "close_prices.parquet").sort_index()
    market_data = pd.read_parquet(base_dir / "market_data.parquet").sort_index()
    regime_data = pd.read_parquet(base_dir / "regime_data.parquet").sort_index()
    network_metrics = pd.read_parquet(base_dir / "network_metrics.parquet").sort_index()
    existing_features = pd.read_parquet(base_dir / "network_features.parquet").sort_index()
    existing_sector_features = pd.read_parquet(base_dir / "sector_centrality_features.parquet").sort_index()

    sector_dict = dict(zip(sector_mapping["Ticker"], sector_mapping["Sector"]))
    tickers = sector_mapping["Ticker"].tolist()

    with open(base_dir / "networks" / "correlation_matrices_pearson.pkl", "rb") as fh:
        import pickle

        existing_corr = pickle.load(fh)
    with open(base_dir / "networks" / "node_centralities.pkl", "rb") as fh:
        existing_node_centralities = pickle.load(fh)

    last_price_date = pd.Timestamp(close_prices.index.max()).normalize()
    target_end = pd.Timestamp(end_date).normalize()
    if target_end <= last_price_date:
        raise ValueError(f"Target end date {target_end.date()} is not after current max date {last_price_date.date()}.")

    symbols = tickers + ["^GSPC", "^VIX"]
    delta_prices = _download_close_prices(symbols, start_date=last_price_date, end_date=target_end)
    delta_prices = delta_prices.loc[delta_prices.index > last_price_date]
    if delta_prices.empty:
        raise ValueError("Download succeeded but returned no newer rows.")

    missing_tickers = [ticker for ticker in tickers if ticker not in delta_prices.columns]
    for ticker in missing_tickers:
        delta_prices[ticker] = pd.NA
    if missing_tickers:
        print(
            f"Warning: {len(missing_tickers)} ticker(s) missing from Yahoo download; "
            "applying delisting/inactive policy for them if configured:",
            ", ".join(missing_tickers[:10]),
        )

    delta_prices = delta_prices.reindex(columns=symbols)
    for symbol in ("^GSPC", "^VIX"):
        if symbol not in delta_prices.columns:
            raise ValueError(f"Missing downloaded market series: {symbol}")

    extended_close = _combine_history(close_prices[tickers], delta_prices[tickers])
    extended_close, inactive_summary = _apply_delisting_policy(extended_close)
    returns_simple = extended_close.pct_change(fill_method=None).fillna(0.0)
    returns_simple.index.name = "Date"

    sp500_close = _combine_history(
        market_data[["SP500"]].rename(columns={"SP500": "^GSPC"}),
        delta_prices[["^GSPC"]],
    )["^GSPC"].ffill()
    vix_close = _combine_history(
        market_data[["VIX"]].rename(columns={"VIX": "^VIX"}),
        delta_prices[["^VIX"]],
    )["^VIX"].ffill()
    extended_market_data, extended_regime_data = build_regime_frame(returns_simple, sp500_close, vix_close)

    all_corr, all_node_centralities, new_feature_core, new_sector_features, new_metrics = compute_incremental_network_outputs(
        returns_simple=returns_simple,
        market_data=extended_market_data,
        regime_data=extended_regime_data,
        sector_dict=sector_dict,
        existing_corr_matrices=existing_corr,
        existing_node_centralities=existing_node_centralities,
        threshold=threshold,
        window=window,
    )

    if new_metrics.empty:
        raise ValueError("No new network snapshots were generated. Check the date range and rolling window.")

    all_network_metrics = pd.concat([network_metrics, new_metrics]).sort_index()
    all_feature_core = pd.concat([existing_features, new_feature_core], sort=False).sort_index()
    all_sector_features = pd.concat([existing_sector_features, new_sector_features], sort=False).sort_index()
    all_network_features = build_network_features(all_feature_core, all_sector_features)

    write_processed_dataset(
        output_dir=output_dir,
        sector_mapping=sector_mapping,
        close_prices=extended_close,
        returns_simple=returns_simple,
        market_data=extended_market_data,
        regime_data=extended_regime_data,
        network_metrics=all_network_metrics,
        network_features=all_network_features,
        sector_centrality_features=all_sector_features,
        correlation_matrices=all_corr,
        node_centralities=all_node_centralities,
    )

    report = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "price_date_min": str(extended_close.index.min().date()),
        "price_date_max": str(extended_close.index.max().date()),
        "snapshot_date_min": str(min(all_corr).date()),
        "snapshot_date_max": str(max(all_corr).date()),
        "added_price_rows": int(len(extended_close.index.difference(close_prices.index))),
        "added_snapshot_rows": int(len(new_metrics)),
        "threshold": threshold,
        "window": window,
        "inactive_tickers": inactive_summary,
    }
    (output_dir / "refresh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Extend RiskSentinel processed data with newer market rows.")
    parser.add_argument("--base-dir", default=str(data_loader.FINAL), help="Existing processed dataset directory.")
    parser.add_argument(
        "--output-dir",
        default="data/processed_extended",
        help="Directory where the extended processed dataset will be written.",
    )
    parser.add_argument("--end-date", default="2026-02-28", help="Target end date (YYYY-MM-DD).")
    parser.add_argument("--threshold", type=float, default=0.3, help="Absolute correlation threshold for edges.")
    parser.add_argument("--window", type=int, default=60, help="Rolling correlation window in trading days.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report = extend_processed_dataset(
        base_dir=base_dir,
        output_dir=output_dir,
        end_date=args.end_date,
        threshold=args.threshold,
        window=args.window,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
