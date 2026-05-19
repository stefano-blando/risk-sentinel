#!/usr/bin/env python3
"""Run a fixed-origin systemic-risk forecast backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import data_loader
from src.core.forecasting import (
    build_direct_feature_frame,
    build_forecast_frame,
    run_full_evaluation_on_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run systemic-risk outlook backtests.")
    parser.add_argument("--train-end", default="2025-11-30", help="Training cutoff date.")
    parser.add_argument("--test-end", default="2026-02-28", help="Final backtest date.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty.")
    parser.add_argument("--walk-step-days", type=int, default=20, help="Step size for last-year walk-forward folds.")
    parser.add_argument("--walk-horizon-days", type=int, default=20, help="Forecast horizon for walk-forward folds.")
    parser.add_argument("--output-json", default="artifacts/systemic_risk_forecast_latest.json", help="JSON report path.")
    parser.add_argument("--output-csv", default="artifacts/systemic_risk_forecast_latest.csv", help="CSV detail path.")
    args = parser.parse_args()

    network_metrics = data_loader.load_network_metrics()
    network_features = data_loader.load_network_features()
    regime_data = data_loader.load_regime_data()
    frame = build_forecast_frame(network_metrics, regime_data)
    direct_feature_frame = build_direct_feature_frame(network_features, regime_data)
    node_centralities = data_loader.load_node_centralities()
    report, joined = run_full_evaluation_on_frame(
        frame,
        node_centralities,
        direct_feature_frame=direct_feature_frame,
        train_end=args.train_end,
        test_end=args.test_end,
        alpha=args.alpha,
        walk_step_days=args.walk_step_days,
        walk_horizon_days=args.walk_horizon_days,
    )

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    if not out_json.is_absolute():
        out_json = (ROOT / out_json).resolve()
    if not out_csv.is_absolute():
        out_csv = (ROOT / out_csv).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joined.to_csv(out_csv)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
