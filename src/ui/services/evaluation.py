"""Evaluation helpers for benchmark and routing packs."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def run_local_benchmark(
    *,
    benchmark_queries: list[str],
    parse_chat_query_fn,
    infer_model_from_query_fn,
    do_build_network_fn,
    contagion_module,
    snapshot_local_state_fn,
    restore_local_state_fn,
    session_state,
    np_module,
    threshold: float,
) -> dict:
    """Quick benchmark pack for hackathon rehearsals (local deterministic)."""
    preserved = snapshot_local_state_fn(session_state)
    rows = []
    t0 = time.perf_counter()
    ok = 0
    try:
        for query in benchmark_queries:
            q_start = time.perf_counter()
            parsed = parse_chat_query_fn(query)
            if not parsed:
                rows.append(
                    {
                        "query": query,
                        "status": "parse_failed",
                        "latency_s": round(time.perf_counter() - q_start, 3),
                    }
                )
                continue

            model = infer_model_from_query_fn(query)
            date = parsed.get("date") or "2025-12-01"
            try:
                graph = do_build_network_fn(date, threshold, emit_messages=False)
                result = contagion_module.run_shock_scenario(
                    graph,
                    parsed["ticker"],
                    parsed["shock"] / 100.0,
                    model,
                )
                summary = result.summary()
                ok += 1
                rows.append(
                    {
                        "query": query,
                        "status": "ok",
                        "ticker": parsed["ticker"],
                        "shock": parsed["shock"],
                        "model": model,
                        "waves": summary["cascade_depth"],
                        "affected": summary["n_affected"],
                        "avg_stress_pct": round(summary["avg_stress"] * 100, 2),
                        "latency_s": round(time.perf_counter() - q_start, 3),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "query": query,
                        "status": f"err:{type(exc).__name__}",
                        "latency_s": round(time.perf_counter() - q_start, 3),
                    }
                )
    finally:
        restore_local_state_fn(session_state, preserved)

    total = time.perf_counter() - t0
    latencies = [row["latency_s"] for row in rows]
    return {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(benchmark_queries),
        "n_ok": ok,
        "success_rate_pct": round((ok / max(1, len(benchmark_queries))) * 100, 1),
        "avg_latency_s": round(float(np_module.mean(latencies)) if latencies else 0.0, 3),
        "total_time_s": round(total, 3),
        "rows": rows,
    }


def run_scenario_pack_eval(
    *,
    scenario_pack: list[dict],
    parse_chat_query_fn,
    is_query_in_scope_fn,
    is_complex_query_fn,
    choose_execution_policy_fn,
    session_state,
    access_allowed: bool,
) -> dict:
    """Evaluate routing expectations for curated judge scenario pack."""
    rows = []
    for scenario in scenario_pack:
        query = scenario["query"]
        parsed = parse_chat_query_fn(query)
        in_scope, _ = is_query_in_scope_fn(query, parsed)
        complex_query = is_complex_query_fn(query)
        policy = choose_execution_policy_fn(
            parsed=parsed,
            complex_query=complex_query,
            in_scope=in_scope,
            agent_mode=session_state.agent_mode,
            gpt_for_parseable_queries=session_state.gpt_for_parseable_queries,
            access_allowed=access_allowed,
            selected_strategy=session_state.agent_strategy,
        )
        actual = policy.get("route", "n/a")
        expected = scenario["expected_route"]
        ok = expected in actual or (expected == "gpt" and actual == "gpt")
        rows.append(
            {
                "scenario": scenario["name"],
                "expected_route": expected,
                "actual_route": actual,
                "effective_strategy": policy.get("effective_strategy"),
                "status": "PASS" if ok else "CHECK",
            }
        )

    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    return {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": len(rows),
        "n_pass": pass_count,
        "pass_rate_pct": round(pass_count / max(1, len(rows)) * 100, 1),
        "rows": rows,
    }
