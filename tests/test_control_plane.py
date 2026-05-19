from src.agents.control_plane import EvidenceLedger, MemoryStore, WorkflowStateMachine


def test_state_machine_valid_path():
    sm = WorkflowStateMachine()
    assert sm.state == "received"
    sm.transition("local_facts")
    sm.transition("analysis")
    sm.transition("critic")
    sm.transition("finalize")
    assert sm.state == "finalize"


def test_state_machine_invalid_transition():
    sm = WorkflowStateMachine()
    sm.transition("local_facts")
    try:
        sm.transition("critic")
        assert False, "expected invalid transition error"
    except RuntimeError as exc:
        assert "Invalid transition" in str(exc)


def test_evidence_ledger_incremental_ids():
    ledger = EvidenceLedger()
    e1 = ledger.add(source="a", kind="k", content="first")
    e2 = ledger.add(source="b", kind="k", content="second")
    assert e1.evidence_id == "E1"
    assert e2.evidence_id == "E2"
    assert ledger.ids() == ["E1", "E2"]


def test_memory_store_semantic_invalidation_by_regime_and_data_tag():
    mem = MemoryStore(ttl_sec=3600, max_items=10)
    mem.put_semantic("k1", "value", regime="Calm", data_tag="v1")

    assert mem.get_semantic("k1", regime="Calm", data_tag="v1") == "value"
    assert mem.get_semantic("k1", regime="Crisis", data_tag="v1") == ""
    assert mem.get_semantic("k1", regime="Calm", data_tag="v2") == ""
