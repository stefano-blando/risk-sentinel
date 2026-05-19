#!/usr/bin/env python3
"""Smoke-check core deterministic demo scenarios used in the hackathon pitch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import contagion, data_loader, network


@dataclass(frozen=True)
class Scenario:
    name: str
    date: str
    ticker: str
    shock_pct: int
    threshold: float
    model: str = "debtrank"


SCENARIOS: list[Scenario] = [
    Scenario("COVID-19 Crash", "2020-03-16", "BAC", 60, 0.75),
    Scenario("SVB Crisis", "2023-03-13", "SCHW", 50, 0.60),
    Scenario("Japan Carry Trade", "2024-08-05", "GS", 40, 0.60),
    Scenario("Volmageddon 2018", "2018-02-08", "JPM", 40, 0.65),
    Scenario("Russia-Ukraine", "2022-03-01", "XOM", 50, 0.65),
]


def _run_scenario(s: Scenario, sector_dict: dict[str, str]) -> dict:
    t0 = time.perf_counter()
    corr, actual_date = data_loader.get_correlation_matrix(s.date)
    graph = network.build_network(corr, threshold=s.threshold, sector_dict=sector_dict)
    metrics = network.compute_global_metrics(graph)

    if s.ticker not in graph:
        raise ValueError(f"Ticker `{s.ticker}` not found in graph for {actual_date.date()}.")

    result = contagion.run_shock_scenario(graph, s.ticker, s.shock_pct / 100.0, s.model)
    summary = result.summary()

    n_nodes = int(graph.number_of_nodes())
    avg_stress = float(summary["avg_stress"])
    if n_nodes <= 0:
        raise ValueError("Empty graph produced.")
    if not (0.0 <= avg_stress <= 1.0):
        raise ValueError(f"Invalid avg_stress={avg_stress}.")
    if int(summary["cascade_depth"]) < 0:
        raise ValueError("Negative cascade depth.")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "scenario": s.name,
        "input": {
            "date": s.date,
            "ticker": s.ticker,
            "shock_pct": s.shock_pct,
            "threshold": s.threshold,
            "model": s.model,
        },
        "actual_date": str(actual_date.date()),
        "network": {
            "n_nodes": int(metrics.get("n_nodes", 0)),
            "n_edges": int(metrics.get("n_edges", 0)),
            "density": round(float(metrics.get("density", 0.0)), 6),
        },
        "summary": summary,
        "elapsed_ms": elapsed_ms,
        "status": "pass",
    }


def run_demo_check() -> dict:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sector_dict = data_loader.get_sector_dict()
    rows: list[dict] = []
    failures: list[dict] = []

    for scenario in SCENARIOS:
        try:
            row = _run_scenario(scenario, sector_dict)
            rows.append(row)
            print(
                f"[PASS] {scenario.name}: {row['summary']['n_affected']} affected, "
                f"depth {row['summary']['cascade_depth']}, {row['elapsed_ms']} ms"
            )
        except Exception as exc:  # noqa: BLE001
            fail = {
                "scenario": scenario.name,
                "status": "fail",
                "error": str(exc),
            }
            failures.append(fail)
            rows.append(fail)
            print(f"[FAIL] {scenario.name}: {exc}")

    n_total = len(SCENARIOS)
    n_fail = len(failures)
    n_pass = n_total - n_fail
    return {
        "started_at_utc": started,
        "n_scenarios": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_rate_pct": round((n_pass / n_total) * 100.0, 2) if n_total else 0.0,
        "ok": n_fail == 0,
        "rows": rows,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic smoke checks for demo scenarios.")
    parser.add_argument(
        "--output",
        default="artifacts/demo_check_latest.json",
        help="Path to JSON output file (default: artifacts/demo_check_latest.json)",
    )
    args = parser.parse_args()

    payload = run_demo_check()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote demo check report: {out_path}")
    print(f"Result: {payload['n_pass']}/{payload['n_scenarios']} passed ({payload['pass_rate_pct']}%)")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
