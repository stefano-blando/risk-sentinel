"""Reporting helpers for RiskSentinel action packs and JSON-safe payloads."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def json_safe(value: Any):
    """Recursively convert runtime objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, nx.Graph):
        n_nodes = int(value.number_of_nodes())
        n_edges = int(value.number_of_edges())
        density = float(nx.density(value)) if n_nodes > 0 else 0.0
        return {
            "_type": "networkx.Graph",
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": round(density, 6),
        }
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def generate_action_pack_ceo_brief(
    *,
    graph_data: dict | None,
    shock_summary: dict | None,
    commander: dict | None,
    autonomous: dict | None,
    portfolio: dict | None,
) -> str:
    """One-page CEO narrative for fast executive decisioning."""
    if not graph_data and not commander and not autonomous:
        return "# CEO Brief\n\nNo actionable scenario available yet."

    lines = ["# RiskSentinel CEO Action Brief", ""]
    if graph_data:
        lines.append(f"- **Market regime**: {graph_data['regime']} (VIX {graph_data['vix']:.1f}) on {graph_data['date']}")
    if shock_summary:
        lines.append(
            f"- **Primary scenario**: {shock_summary['shocked_node']} {shock_summary['shock_magnitude']*100:.0f}% shock "
            f"-> {shock_summary['n_affected']} affected, avg stress {shock_summary['avg_stress']*100:.1f}%"
        )
    top_pick = (commander or {}).get("top_pick")
    if top_pick:
        lines.append(
            f"- **Commander top risk**: {top_pick['ticker']} ({top_pick['sector']}) "
            f"score {top_pick['risk_score']:.1f}, depth {top_pick['cascade_depth']} waves"
        )
    auto_rows = (autonomous or {}).get("rows") or []
    if auto_rows:
        top_auto = auto_rows[0]
        lines.append(
            f"- **Autonomous fragility**: {top_auto['ticker']} @ {top_auto['shock_pct']}% "
            f"score {top_auto['risk_score']:.1f}"
        )
    if (portfolio or {}).get("ok"):
        lines.append(
            f"- **Portfolio lens**: expected stress {portfolio.get('expected_stress_pct', 0.0):.1f}% "
            f"-> {portfolio.get('expected_stress_pct_after_hedge', 0.0):.1f}% after hedges "
            f"(~{portfolio.get('estimated_loss_avoided_pct', 0.0):.1f}% avoided)."
        )
    lines.extend(
        [
            "",
            "## Immediate Actions (Next 24h)",
            "1. Cap exposure to top contagion hubs identified by Commander.",
            "2. Activate sector hedges where concentration and contagion overlap.",
            "3. Monitor cascade depth and VIX acceleration as hard escalation triggers.",
        ]
    )
    return "\n".join(lines)


def generate_action_pack_runbook(*, commander: dict | None, portfolio: dict | None) -> str:
    """Operational runbook for risk desk execution."""
    lines = [
        "# RiskSentinel Risk Desk Runbook",
        "",
        "## Step 1 — Validate Inputs",
        "1. Confirm date/threshold/model and shock settings.",
        "2. Confirm data freshness and selected risk profile.",
        "",
        "## Step 2 — Scenario Commander",
        "1. Run Commander and capture top-3 systemic seeds.",
        "2. Execute focused shock sims on top seed + correlated peers.",
        "",
        "## Step 3 — Autonomous Sweep",
        "1. Run autonomous stress grid (30/50/70%).",
        "2. Flag hidden fragilities with high score at <=50% shock.",
        "",
        "## Step 4 — Portfolio Co-Pilot",
        "1. Load current exposures (ticker, weight).",
        "2. Compute weighted contagion risk and sector concentration.",
        "3. Apply hedge actions and estimate avoided stress.",
        "",
        "## Step 5 — Governance",
        "1. Keep critic gate ON (auto-repair enabled).",
        "2. Export action pack + explainability trace.",
    ]
    top = (commander or {}).get("top_pick")
    if top:
        lines.extend(
            [
                "",
                "## Current Top Risk",
                f"- {top['ticker']} ({top['sector']}) | score {top['risk_score']:.1f} | depth {top['cascade_depth']}",
            ]
        )
    if (portfolio or {}).get("ok"):
        lines.extend(["", "## Portfolio Actions"] + [f"- {a}" for a in portfolio.get("actions", [])[:6]])
    return "\n".join(lines)


def build_action_pack_payload(
    *,
    generated_at_utc: str,
    market_context: dict | None,
    commander: dict | None,
    autonomous_stress_test: dict | None,
    portfolio_copilot: dict | None,
    trace_summary: dict | None,
    policy_plan: list,
    executor_log: list,
    session_memory: list,
) -> dict:
    return {
        "schema_version": "action_pack.v1",
        "generated_at_utc": generated_at_utc,
        "market_context": market_context,
        "commander": commander,
        "autonomous_stress_test": autonomous_stress_test,
        "portfolio_copilot": portfolio_copilot,
        "trace_summary": trace_summary,
        "policy_plan": policy_plan,
        "executor_log": executor_log,
        "session_memory": session_memory,
    }


def generate_action_pack_machine_json(payload: dict) -> str:
    return json.dumps(json_safe(payload), indent=2)

