"""Run trace helpers."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def create_run_trace(
    *,
    query: str,
    parsed: dict | None,
    complex_query: bool,
    in_scope: bool,
    scope_reason: str,
    model_for_query: str,
    threshold: float,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": f"run-{int(now.timestamp() * 1000)}",
        "created_at_utc": now.isoformat(),
        "query": query,
        "parsed": parsed or {},
        "complex_query": complex_query,
        "in_scope": in_scope,
        "scope_reason": scope_reason,
        "model": model_for_query,
        "threshold": threshold,
        "policy": {},
        "timings": {},
        "events": [],
        "result": {},
        "workflow": {"state": "received", "history": ["received"]},
    }


def trace_event(trace: dict, label: str, detail: str = "") -> None:
    trace["events"].append(
        {
            "t_sec": round(time.perf_counter() - trace["_t0"], 3),
            "label": label,
            "detail": detail,
        }
    )


def advance_workflow(trace: dict, next_state: str, workflow_transitions: dict[str, set[str]]) -> None:
    wf = trace.setdefault("workflow", {"state": "received", "history": ["received"]})
    cur = wf.get("state", "received")
    allowed = workflow_transitions.get(cur, set())
    if next_state in allowed or cur == next_state:
        wf["state"] = next_state
        wf.setdefault("history", []).append(next_state)
        trace_event(trace, "workflow_state", f"{cur}->{next_state}")
    else:
        wf.setdefault("history", []).append(f"invalid:{cur}->{next_state}")
        trace_event(trace, "workflow_invalid_transition", f"{cur}->{next_state}")


def finalize_run_trace(trace: dict, session_state) -> dict:
    trace.pop("_t0", None)
    trace["events"] = trace.get("events", [])[-30:]
    session_state.run_trace = trace
    history = session_state.run_trace_history
    history.append(trace)
    session_state.run_trace_history = history[-50:]
    return trace


def persist_run_trace(trace: dict, *, session_state, app_file: str) -> None:
    """Append trace to local JSONL for post-demo analysis."""
    if not session_state.persist_trace_logs:
        return
    try:
        log_dir = Path(app_file).resolve().parents[1] / "artifacts"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "run_traces.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=True) + "\n")
    except Exception:
        pass
