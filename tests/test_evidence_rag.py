from src.agents.evidence_rag import (
    build_crisis_evidence_docs,
    build_history_evidence_docs,
    format_evidence_block,
    retrieve_evidence,
)


def test_build_crisis_evidence_docs_not_empty():
    events = {
        "COVID-19 Crash": ("2020-02-20", "2020-04-15"),
        "SVB Crisis": ("2023-03-10", "2023-03-20"),
    }
    docs = build_crisis_evidence_docs(events)
    assert len(docs) == 2
    assert docs[0].doc_id.startswith("C")


def test_build_history_evidence_docs_reads_runs():
    history = [
        {
            "query": "What happens if JPM crashes 40%?",
            "result": {"state": "gpt_ok"},
            "timings": {"total_sec": 4.2},
            "quality": {"factual_consistency": True, "uncertainty_score": 0.2},
        }
    ]
    docs = build_history_evidence_docs(history)
    assert len(docs) == 1
    assert "gpt_ok" in docs[0].text


def test_retrieve_evidence_prefers_relevant_docs():
    crisis_docs = build_crisis_evidence_docs(
        {
            "COVID-19 Crash": ("2020-02-20", "2020-04-15"),
            "SVB Crisis": ("2023-03-10", "2023-03-20"),
        }
    )
    out = retrieve_evidence("Compare JPM risk with SVB-like crisis dynamics", crisis_docs, top_k=2)
    assert out
    assert out[0].reference_id == "R1"
    assert any("SVB" in row.title for row in out)


def test_format_evidence_block_contains_refs():
    crisis_docs = build_crisis_evidence_docs({"SVB Crisis": ("2023-03-10", "2023-03-20")})
    out = retrieve_evidence("SVB crisis", crisis_docs, top_k=1)
    text = format_evidence_block(out)
    assert "[R1]" in text
    assert "Retrieved evidence" in text
