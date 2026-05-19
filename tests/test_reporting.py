import json
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd

from src import reporting


def test_json_safe_handles_graph_dataframe_numpy_datetime():
    graph = nx.Graph()
    graph.add_edge("AAA", "BBB", weight=0.7)
    payload = {
        "graph": graph,
        "df": pd.DataFrame([{"ticker": "AAA", "score": np.float64(1.2)}]),
        "n": np.int64(7),
        "ts": pd.Timestamp("2025-12-01"),
    }

    safe = reporting.json_safe(payload)
    assert safe["graph"]["_type"] == "networkx.Graph"
    assert safe["n"] == 7
    assert isinstance(safe["df"], list)
    assert isinstance(safe["ts"], str)
    json.dumps(safe)


def test_generate_action_pack_machine_json_serializes_graph():
    graph = nx.Graph()
    graph.add_edge("AAA", "BBB", weight=0.7)
    payload = reporting.build_action_pack_payload(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        market_context={"G": graph, "date": "2025-12-01"},
        commander={"top_pick": {"ticker": "AAA", "risk_score": 77.5, "sector": "Tech", "cascade_depth": 2}},
        autonomous_stress_test={},
        portfolio_copilot={},
        trace_summary={},
        policy_plan=[],
        executor_log=[],
        session_memory=[],
    )
    raw = reporting.generate_action_pack_machine_json(payload)
    out = json.loads(raw)
    assert out["market_context"]["G"]["_type"] == "networkx.Graph"
    assert out["schema_version"] == "action_pack.v1"


def test_generate_action_pack_ceo_brief_contains_top_risk_line():
    brief = reporting.generate_action_pack_ceo_brief(
        graph_data={"regime": "Crisis", "vix": 29.4, "date": "2025-12-01"},
        shock_summary={
            "shocked_node": "AAA",
            "shock_magnitude": 0.5,
            "n_affected": 12,
            "avg_stress": 0.21,
        },
        commander={"top_pick": {"ticker": "BBB", "sector": "Financials", "risk_score": 81.2, "cascade_depth": 3}},
        autonomous={"rows": [{"ticker": "CCC", "shock_pct": 50, "risk_score": 72.0}]},
        portfolio={"ok": True, "expected_stress_pct": 12.0, "expected_stress_pct_after_hedge": 7.0, "estimated_loss_avoided_pct": 5.0},
    )
    assert "RiskSentinel CEO Action Brief" in brief
    assert "Commander top risk" in brief
