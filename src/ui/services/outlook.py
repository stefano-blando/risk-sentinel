"""Service helpers for the Outlook tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.core import data_loader
from src.core.forecasting import (
    build_direct_feature_frame,
    build_forecast_frame,
    run_backtest_models_on_frame,
    run_full_evaluation_on_frame,
)


REGIME_LABELS = {
    0: "Calm",
    1: "Normal",
    2: "Elevated",
    3: "High",
    4: "Crisis",
}

OUTLOOK_SCENARIOS = [
    {
        "key": "bank",
        "title": "Bank Shock",
        "copy": "Stress a major financial node and inspect second-order spillovers across the market network.",
        "ticker": "JPM",
        "shock_pct": 40,
        "model": "debtrank",
        "threshold": 0.50,
    },
    {
        "key": "tech",
        "title": "Tech Shock",
        "copy": "Probe concentration risk through a large-cap technology shock with broad correlation channels.",
        "ticker": "NVDA",
        "shock_pct": 35,
        "model": "debtrank",
        "threshold": 0.45,
    },
    {
        "key": "energy",
        "title": "Energy Shock",
        "copy": "Test commodity-linked contagion with a sector shock and compare cascade depth against financials.",
        "ticker": "XOM",
        "shock_pct": 30,
        "model": "linear_threshold",
        "threshold": 0.50,
    },
]


def summary_rows_from_forecast(report: dict | None) -> list[dict[str, object]]:
    if not report:
        return []
    rows: list[dict[str, object]] = []
    fixed = report.get("fixed_origin") or {}
    if fixed:
        rows.append(
            {
                "View": "Fixed origin",
                "Model": fixed.get("best_model", fixed.get("model", "n/a")),
                "Regime Acc": fixed.get("regime", {}).get("accuracy"),
                "Density MAE": fixed.get("density", {}).get("mae"),
                "Risk Pressure MAE": fixed.get("risk_pressure", {}).get("mae"),
                "Top-5 Overlap": fixed.get("top_systemic_nodes_persistence", {}).get("mean_overlap"),
            }
        )
    walk = (report.get("walk_forward_last_year") or {}).get("summary") or {}
    if walk:
        rows.append(
            {
                "View": "Walk-forward last year",
                "Model": max((walk.get("best_model_counts") or {"n/a": 0}).items(), key=lambda kv: kv[1])[0],
                "Regime Acc": walk.get("regime_accuracy_mean"),
                "Density MAE": walk.get("density_mae_mean"),
                "Risk Pressure MAE": walk.get("risk_pressure_mae_mean"),
                "Top-5 Overlap": walk.get("top_k_overlap_mean"),
            }
        )
    stress = (report.get("historical_stress_folds") or {}).get("summary") or {}
    if stress:
        rows.append(
            {
                "View": "Historical stress folds",
                "Model": max((stress.get("best_model_counts") or {"n/a": 0}).items(), key=lambda kv: kv[1])[0],
                "Regime Acc": stress.get("regime_accuracy_mean"),
                "Density MAE": stress.get("density_mae_mean"),
                "Risk Pressure MAE": stress.get("risk_pressure_mae_mean"),
                "Top-5 Overlap": stress.get("top_k_overlap_mean"),
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def get_forecast_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    frame = build_forecast_frame(data_loader.load_network_metrics(), data_loader.load_regime_data())
    return pd.Timestamp(frame.index.min()), pd.Timestamp(frame.index.max())


@st.cache_data(show_spinner=False)
def run_live_outlook_cached(
    train_end: str,
    test_end: str,
    alpha: float,
    walk_step_days: int,
    walk_horizon_days: int,
) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame]]:
    frame = build_forecast_frame(data_loader.load_network_metrics(), data_loader.load_regime_data())
    direct_feature_frame = build_direct_feature_frame(
        data_loader.load_network_features(),
        data_loader.load_regime_data(),
    )
    node_centralities = data_loader.load_node_centralities()
    report, joined = run_full_evaluation_on_frame(
        frame,
        node_centralities,
        direct_feature_frame=direct_feature_frame,
        train_end=train_end,
        test_end=test_end,
        alpha=alpha,
        walk_step_days=walk_step_days,
        walk_horizon_days=walk_horizon_days,
        include_direct_model=False,
    )
    _, joined_by_model, best_name = run_backtest_models_on_frame(
        frame,
        direct_feature_frame=direct_feature_frame,
        train_end=train_end,
        test_end=test_end,
        alpha=alpha,
        include_direct_model=False,
    )
    joined_by_model["best"] = joined_by_model[best_name]
    return report, joined, joined_by_model


def regime_label(value: float | int | None) -> str:
    if value is None:
        return "Unknown"
    try:
        if not np.isfinite(float(value)):
            return "Unknown"
        return REGIME_LABELS.get(int(round(float(value))), "Unknown")
    except Exception:
        return "Unknown"


def safe_pct_delta(current: float, previous: float) -> float:
    if abs(previous) < 1e-9:
        return 0.0
    return (current - previous) / abs(previous)


def risk_trend_label(delta_pct: float) -> str:
    if delta_pct >= 0.15:
        return "Rising"
    if delta_pct <= -0.15:
        return "Cooling"
    return "Stable"


def stress_readiness_label(regime_name: str, risk_pressure: float, delta_pct: float) -> str:
    if regime_name in {"High", "Crisis"} or risk_pressure >= 0.22 or delta_pct >= 0.20:
        return "High Alert"
    if regime_name == "Elevated" or risk_pressure >= 0.14 or delta_pct >= 0.08:
        return "Watchlist"
    return "Ready"


def compute_outlook_snapshot(joined: pd.DataFrame, focus_date: str) -> dict[str, object]:
    if joined is None or joined.empty:
        return {}
    focus_ts = pd.Timestamp(focus_date)
    if focus_ts not in joined.index:
        focus_ts = joined.index[-1]
    current_pos = joined.index.get_loc(focus_ts)
    prev_pos = max(0, current_pos - 20)
    prev_ts = joined.index[prev_pos]
    current = joined.loc[focus_ts]
    previous = joined.loc[prev_ts]
    current_regime_label = regime_label(current.get("regime_numeric"))
    previous_regime_label = regime_label(previous.get("regime_numeric"))
    risk_pressure = float(current.get("risk_pressure", 0.0))
    prev_risk_pressure = float(previous.get("risk_pressure", risk_pressure))
    risk_delta_pct = safe_pct_delta(risk_pressure, prev_risk_pressure)
    return {
        "focus_date": str(focus_ts.date()),
        "previous_date": str(prev_ts.date()),
        "regime_label": current_regime_label,
        "prev_regime_label": previous_regime_label,
        "regime_numeric": float(current.get("regime_numeric", np.nan)),
        "prev_regime_numeric": float(previous.get("regime_numeric", np.nan)),
        "risk_pressure": risk_pressure,
        "prev_risk_pressure": prev_risk_pressure,
        "risk_delta_pct": risk_delta_pct,
        "fragility_trend": risk_trend_label(risk_delta_pct),
        "stress_readiness": stress_readiness_label(current_regime_label, risk_pressure, risk_delta_pct),
        "density": float(current.get("density", 0.0)),
        "avg_abs_weight": float(current.get("avg_abs_weight", 0.0)),
        "avg_clustering": float(current.get("avg_clustering", 0.0)),
        "pred_density": float(current.get("pred_density", np.nan)) if "pred_density" in joined.columns else np.nan,
    }


def build_change_rows(joined: pd.DataFrame, focus_date: str) -> list[dict[str, object]]:
    if joined is None or joined.empty:
        return []
    focus_ts = pd.Timestamp(focus_date)
    if focus_ts not in joined.index:
        focus_ts = joined.index[-1]
    current_pos = joined.index.get_loc(focus_ts)
    prev_pos = max(0, current_pos - 20)
    previous = joined.iloc[prev_pos]
    current = joined.loc[focus_ts]
    metric_order = [
        ("density", "Network Density"),
        ("avg_abs_weight", "Avg |Correlation|"),
        ("avg_clustering", "Clustering"),
        ("risk_pressure", "Risk Pressure"),
        ("regime_numeric", "Regime Score"),
    ]
    rows: list[dict[str, object]] = []
    for col, label in metric_order:
        if col not in joined.columns:
            continue
        current_val = float(current[col])
        prev_val = float(previous[col])
        delta = current_val - prev_val
        signal = "Up" if delta > 1e-6 else "Down" if delta < -1e-6 else "Flat"
        rows.append(
            {
                "Metric": label,
                "Current": round(current_val, 4),
                "1M Ago": round(prev_val, 4),
                "Delta": round(delta, 4),
                "Signal": signal,
            }
        )
    return rows


def build_regime_transition_copy(snapshot: dict[str, object]) -> str:
    prev_label = str(snapshot.get("prev_regime_label", "Unknown"))
    curr_label = str(snapshot.get("regime_label", "Unknown"))
    if prev_label == curr_label:
        return f"Regime remains {curr_label.lower()} from {snapshot.get('previous_date')} to {snapshot.get('focus_date')}."
    return f"Regime moved from {prev_label.lower()} to {curr_label.lower()} by {snapshot.get('focus_date')}."


def forecast_confidence_copy(report: dict[str, object], metric: str) -> str:
    fixed = report.get("fixed_origin") or {}
    walk = (report.get("walk_forward_last_year") or {}).get("summary") or {}
    if metric == "regime_numeric":
        acc = float((fixed.get("regime") or {}).get("accuracy", 0.0))
        if acc >= 0.65:
            return "Higher confidence on stable regime monitoring than on crisis transitions."
        return "Low confidence on regime transitions; treat regime forecast as directional only."
    mae_key = {
        "density": "density",
        "avg_abs_weight": "avg_abs_weight",
        "avg_clustering": "avg_clustering",
        "risk_pressure": "risk_pressure",
    }.get(metric, "density")
    mae = float((fixed.get(mae_key) or {}).get("mae", 0.0))
    walk_overlap = float(walk.get("top_k_overlap_mean", 0.0))
    if mae <= 0.03 and walk_overlap >= 0.35:
        return "Useful confidence for aggregate fragility surveillance; limited confidence for exact node ranking."
    return "Useful as a surveillance baseline, but not as a precise crisis predictor."


def build_action_rows(snapshot: dict[str, object], shock_bundle: dict[str, object] | None) -> list[dict[str, object]]:
    readiness = str(snapshot.get("stress_readiness", "Ready"))
    return [
        {
            "Action": "Monitor",
            "Priority": "High" if readiness != "Ready" else "Medium",
            "Rationale": f"Track regime and fragility trend around {snapshot.get('focus_date')}.",
        },
        {
            "Action": "Hedge",
            "Priority": "High" if shock_bundle and shock_bundle["result"].n_affected >= 10 else "Medium",
            "Rationale": "Protect exposures around the most systemic connector and the shocked sector.",
        },
        {
            "Action": "Investigate",
            "Priority": "High" if shock_bundle and shock_bundle["result"].cascade_depth >= 3 else "Medium",
            "Rationale": "Review cross-sector links driving second-order contagion.",
        },
    ]


def build_why_this_matters_rows() -> list[dict[str, object]]:
    return [
        {"Signal": "Current regime", "Why it matters": "Tells supervisors whether the market backdrop is calm or already stressed."},
        {"Signal": "Fragility trend", "Why it matters": "Shows whether network vulnerability is building before a visible crisis."},
        {"Signal": "Forward stress readiness", "Why it matters": "Frames whether a new shock is likely to stay local or spill over."},
        {"Signal": "Top vulnerable nodes", "Why it matters": "Highlights where intervention or hedging attention should go first."},
    ]


def bundle_summary_row(bundle: dict[str, object] | None, label: str) -> dict[str, object]:
    if not bundle:
        return {
            "Scenario": label,
            "Ticker": "n/a",
            "Model": "n/a",
            "Policy": "n/a",
            "Affected": 0,
            "Waves": 0,
            "Avg Stress %": 0.0,
            "Total Stress": 0.0,
        }
    summary = bundle["result"].summary()
    return {
        "Scenario": label,
        "Ticker": bundle.get("ticker", "n/a"),
        "Model": bundle.get("model", "n/a"),
        "Policy": (bundle.get("intervention_meta") or {}).get("label", "No intervention"),
        "Affected": int(summary["n_affected"]),
        "Waves": int(summary["cascade_depth"]),
        "Avg Stress %": round(100.0 * float(summary["avg_stress"]), 1),
        "Total Stress": round(float(summary["total_stress"]), 3),
    }


def build_compare_rows(primary: dict[str, object] | None, secondary: dict[str, object] | None) -> list[dict[str, object]]:
    return [
        bundle_summary_row(primary, "Primary"),
        bundle_summary_row(secondary, "Comparison"),
    ]


def build_counterfactual_row(
    primary: dict[str, object] | None,
    secondary: dict[str, object] | None,
) -> dict[str, object] | None:
    if not primary or not secondary:
        return None
    p = primary["result"].summary()
    s = secondary["result"].summary()
    return {
        "Affected delta": int(s["n_affected"]) - int(p["n_affected"]),
        "Wave delta": int(s["cascade_depth"]) - int(p["cascade_depth"]),
        "Avg stress delta %": round(100.0 * (float(s["avg_stress"]) - float(p["avg_stress"])), 1),
        "Total stress delta": round(float(s["total_stress"]) - float(p["total_stress"]), 3),
    }


def build_narrative_lines(
    snapshot: dict[str, object],
    vulnerability_rows: list[dict[str, object]],
    shock_bundle: dict[str, object] | None,
) -> list[str]:
    if not snapshot:
        return []
    lines = [
        f"System state at {snapshot['focus_date']}: {snapshot['regime_label']} regime with {snapshot['fragility_trend'].lower()} fragility trend.",
        f"Risk pressure moved from {snapshot['prev_risk_pressure']:.3f} to {snapshot['risk_pressure']:.3f} over the last month, so readiness is {snapshot['stress_readiness'].lower()}.",
    ]
    if vulnerability_rows:
        top_row = vulnerability_rows[0]
        lines.append(
            f"Most systemically important node in the baseline view is {top_row['Ticker']} in {top_row['Sector']}."
        )
    if shock_bundle:
        summary = shock_bundle["result"].summary()
        lines.append(
            f"Under the selected shock, {summary['shocked_node']} affects {summary['n_affected']} names over {summary['cascade_depth']} waves with average stress {100.0 * float(summary['avg_stress']):.1f}%."
        )
    else:
        lines.append("Run one of the scenario cards below to convert the monitoring view into a forward stress test.")
    return lines[:4]


def top_systemic_rows(
    data_loader_obj,
    sector_dict: dict[str, str] | None,
    date_str: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    centralities, actual_date = data_loader_obj.get_node_centralities_for_date(date_str)
    ranked = sorted(
        centralities.items(),
        key=lambda item: float(item[1].get("pagerank", 0.0)),
        reverse=True,
    )[:limit]
    rows: list[dict[str, object]] = []
    for rank, (ticker, vals) in enumerate(ranked, start=1):
        rows.append(
            {
                "Date": str(actual_date.date()),
                "Rank": rank,
                "Ticker": ticker,
                "Sector": vals.get("Sector", (sector_dict or {}).get(ticker, "Unknown")),
                "Pagerank": round(float(vals.get("pagerank", 0.0)), 4),
            }
        )
    return rows


def build_watchlist_rows(
    data_loader_obj,
    sector_dict: dict[str, str] | None,
    focus_date: str,
    lookback: int = 20,
    limit: int = 8,
) -> list[dict[str, object]]:
    current_c, actual_focus = data_loader_obj.get_node_centralities_for_date(focus_date)
    dates = pd.DatetimeIndex(sorted(data_loader_obj.load_node_centralities().keys()))
    focus_ts = pd.Timestamp(actual_focus)
    if focus_ts not in dates:
        return []
    pos = dates.get_loc(focus_ts)
    prev_ts = dates[max(0, pos - lookback)]
    previous_c, _ = data_loader_obj.get_node_centralities_for_date(str(prev_ts.date()))
    rows: list[dict[str, object]] = []
    for ticker, vals in current_c.items():
        curr = float(vals.get("pagerank", 0.0))
        prev = float(previous_c.get(ticker, {}).get("pagerank", 0.0))
        delta = curr - prev
        if abs(delta) < 1e-6:
            continue
        rows.append(
            {
                "Ticker": ticker,
                "Sector": vals.get("Sector", (sector_dict or {}).get(ticker, "Unknown")),
                "Current PR": round(curr, 4),
                "Prev PR": round(prev, 4),
                "Delta PR": round(delta, 4),
                "Direction": "Rising" if delta > 0 else "Cooling",
            }
        )
    rows.sort(key=lambda row: abs(float(row["Delta PR"])), reverse=True)
    return rows[:limit]


def build_vulnerability_rows(
    data_loader_obj,
    sector_dict: dict[str, str] | None,
    date_str: str,
    shock_bundle: dict[str, object] | None,
    limit: int = 8,
) -> list[dict[str, object]]:
    baseline_rows = top_systemic_rows(data_loader_obj, sector_dict, date_str, limit=max(limit * 4, 20))
    baseline_rank = {row["Ticker"]: row["Rank"] for row in baseline_rows}
    baseline_pr = {row["Ticker"]: row["Pagerank"] for row in baseline_rows}
    baseline_sector = {row["Ticker"]: row["Sector"] for row in baseline_rows}
    if not shock_bundle:
        return [
            {
                "Ticker": row["Ticker"],
                "Sector": row["Sector"],
                "Baseline Rank": row["Rank"],
                "Baseline PR": row["Pagerank"],
                "Shock Impact Rank": None,
                "Shock Stress %": None,
                "Delta Rank": None,
            }
            for row in baseline_rows[:limit]
        ]

    affected = shock_bundle["result"].affected_nodes[:limit]
    impacted_rank = {ticker: idx + 1 for idx, (ticker, _) in enumerate(affected)}
    impacted_stress = {ticker: float(stress) for ticker, stress in affected}
    ordered_tickers = sorted(
        set(baseline_rank) | set(impacted_rank),
        key=lambda ticker: (
            impacted_rank.get(ticker, 10_000),
            baseline_rank.get(ticker, 10_000),
            ticker,
        ),
    )
    rows: list[dict[str, object]] = []
    for ticker in ordered_tickers[:limit]:
        base_rank = baseline_rank.get(ticker)
        stress_rank = impacted_rank.get(ticker)
        delta_rank = None
        if base_rank is not None and stress_rank is not None:
            delta_rank = base_rank - stress_rank
        rows.append(
            {
                "Ticker": ticker,
                "Sector": baseline_sector.get(ticker, (sector_dict or {}).get(ticker, "Unknown")),
                "Baseline Rank": base_rank if base_rank is not None else pd.NA,
                "Baseline PR": baseline_pr.get(ticker, pd.NA),
                "Shock Impact Rank": stress_rank if stress_rank is not None else pd.NA,
                "Shock Stress %": round(100.0 * impacted_stress[ticker], 1) if ticker in impacted_stress else pd.NA,
                "Delta Rank": delta_rank if delta_rank is not None else pd.NA,
            }
        )
    return rows


def build_why_nodes_rows(
    data_loader_obj,
    sector_dict: dict[str, str] | None,
    date_str: str,
    shock_bundle: dict[str, object] | None,
    limit: int = 6,
) -> list[dict[str, object]]:
    if not shock_bundle:
        return []
    centralities, _ = data_loader_obj.get_node_centralities_for_date(date_str)
    affected = shock_bundle["result"].affected_nodes[:limit]
    rows: list[dict[str, object]] = []
    for rank, (ticker, stress) in enumerate(affected, start=1):
        vals = centralities.get(ticker, {})
        reasons = []
        if float(vals.get("pagerank", 0.0)) >= 0.008:
            reasons.append("high pagerank")
        if float(vals.get("degree", 0.0)) >= 0.10:
            reasons.append("dense connectivity")
        sector = vals.get("Sector", (sector_dict or {}).get(ticker, "Unknown"))
        reasons.append(f"{sector} exposure")
        rows.append(
            {
                "Rank": rank,
                "Ticker": ticker,
                "Sector": sector,
                "Stress %": round(100.0 * float(stress), 1),
                "Pagerank": round(float(vals.get("pagerank", 0.0)), 4),
                "Why": ", ".join(reasons),
            }
        )
    return rows
