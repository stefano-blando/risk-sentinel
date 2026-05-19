from src.agents.evidence_validation import validate_payload_evidence


def test_rejects_unknown_rag_reference():
    payload = {
        "situation": ["Stress resembles prior episodes."],
        "quant_results": ["Affected nodes: 124"],
        "actions": ["Trim exposure."],
        "monitoring_triggers": ["Watch VIX > 30."],
        "evidence_used": ["R9"],
        "notes": "Based on R9.",
        "insufficient_data": False,
    }
    out = validate_payload_evidence(
        payload,
        allowed_r_refs={"R1", "R2"},
        require_reference_for_numeric_claims=True,
        facts_available=True,
    )
    assert out["approved"] is False
    assert any("Unknown RAG references" in msg for msg in out["issues"])


def test_rejects_cited_reference_missing_from_evidence_used():
    payload = {
        "situation": ["Signals align with R1 context."],
        "quant_results": ["Avg stress 22.4%"],
        "actions": ["Reduce concentration."],
        "monitoring_triggers": ["Track stress waves."],
        "evidence_used": [],
        "notes": "R1 supports sector spillovers.",
        "insufficient_data": False,
    }
    out = validate_payload_evidence(
        payload,
        allowed_r_refs={"R1"},
        require_reference_for_numeric_claims=True,
        facts_available=True,
    )
    assert out["approved"] is False
    assert any("not listed in evidence_used" in msg for msg in out["issues"])


def test_requires_reference_for_numeric_claims_when_enabled():
    payload = {
        "situation": ["Systemic risk elevated."],
        "quant_results": ["Affected nodes: 140", "Avg stress: 19.2%"],
        "actions": ["Add hedges."],
        "monitoring_triggers": ["Escalate if defaults > 10."],
        "evidence_used": [],
        "notes": "",
        "insufficient_data": False,
    }
    out = validate_payload_evidence(
        payload,
        allowed_r_refs=set(),
        require_reference_for_numeric_claims=True,
        facts_available=True,
    )
    assert out["approved"] is False
    assert any("Numeric claims are present" in msg for msg in out["issues"])


def test_accepts_valid_payload_with_consistent_references():
    payload = {
        "situation": ["Observed contagion channels consistent with R1."],
        "quant_results": ["Affected nodes: 118", "Average stress: 17.5%"],
        "actions": ["Limit concentration in top hubs."],
        "monitoring_triggers": ["Escalate if cascade depth >= 6."],
        "evidence_used": ["E1", "R1"],
        "notes": "R1 adds historical context; E1 covers deterministic run output.",
        "insufficient_data": False,
    }
    out = validate_payload_evidence(
        payload,
        allowed_r_refs={"R1", "R2"},
        allowed_e_refs={"E1", "E2"},
        require_reference_for_numeric_claims=True,
        facts_available=True,
    )
    assert out["approved"] is True
    assert out["issues"] == []

