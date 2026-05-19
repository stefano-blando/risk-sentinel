"""Service helpers for the Audit Trail tab."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.agents.evaluation import EvalSample, evaluate_samples
from src import reporting
from src.ui.services.surveillance import build_severity_df


def summarize_quality(history: list[dict]) -> dict:
    if not history:
        return {}
    eval_rows = [row.get("quality", {}) for row in history if row.get("quality")]
    if not eval_rows:
        return {}
    n_rows = len(eval_rows)
    factual_rows = [row for row in eval_rows if row.get("factual_consistency") is not None]
    factual_ok = sum(1 for row in factual_rows if row.get("factual_consistency"))
    return {
        "runs": n_rows,
        "avg_latency_sec": round(float(np.mean([row.get("latency_sec", 0.0) for row in eval_rows])), 2),
        "cache_hit_rate_pct": round(sum(1 for row in eval_rows if row.get("cache_hit")) / n_rows * 100, 1),
        "fallback_rate_pct": round(sum(1 for row in eval_rows if row.get("used_fallback")) / n_rows * 100, 1),
        "gpt_success_rate_pct": round(sum(1 for row in eval_rows if row.get("gpt_success")) / n_rows * 100, 1),
        "rate_limit_events_total": int(sum(int(row.get("rate_limit_events", 0)) for row in eval_rows)),
        "factual_consistency_pct": round((factual_ok / max(1, len(factual_rows))) * 100, 1) if factual_rows else None,
        "avg_uncertainty": round(float(np.mean([row.get("uncertainty_score", 0.5) for row in eval_rows])), 3),
    }


def build_judge_kpis(history: list[dict]) -> dict:
    samples: list[EvalSample] = []
    for row in history:
        quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        timings = row.get("timings", {}) if isinstance(row.get("timings"), dict) else {}

        critic_approved = result.get("critic_approved")
        if not isinstance(critic_approved, bool):
            critic_approved = bool(result.get("structured_output_valid", False) and result.get("gpt_success", False))

        factual = quality.get("factual_consistency")
        factual_consistent = bool(factual) if factual is not None else False
        latency_sec = float(timings.get("total_sec", 0.0) or 0.0)
        fallback_used = bool(quality.get("used_fallback", False))
        samples.append(
            EvalSample(
                critic_approved=critic_approved,
                factual_consistent=factual_consistent,
                latency_sec=latency_sec,
                fallback_used=fallback_used,
            )
        )

    out = evaluate_samples(samples)
    out["gpt_runs"] = int(sum(1 for row in history if row.get("result", {}).get("gpt_attempted")))
    out["gpt_success_runs"] = int(sum(1 for row in history if row.get("result", {}).get("gpt_success")))
    return out


def build_judge_run_rows(history: list[dict], limit: int = 20) -> pd.DataFrame:
    rows: list[dict] = []
    for row in history[-limit:]:
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
        timings = row.get("timings", {}) if isinstance(row.get("timings"), dict) else {}
        rows.append(
            {
                "query": str(row.get("query", ""))[:90],
                "state": result.get("state", "n/a"),
                "critic_approved": result.get("critic_approved", None),
                "factual_consistency": quality.get("factual_consistency", None),
                "latency_sec": round(float(timings.get("total_sec", 0.0) or 0.0), 2),
                "fallback": bool(quality.get("used_fallback", False)),
                "uncertainty": quality.get("uncertainty_score", None),
            }
        )
    return pd.DataFrame(rows)


def generate_report_text(
    *,
    graph_data: dict | None,
    shock_result,
    sector_dict: dict[str, str] | None,
    agent_messages: list[tuple],
) -> str:
    if not graph_data or not shock_result:
        return "No simulation results to report."

    summary = shock_result.summary()
    lines = [
        "=" * 60,
        "RISKSENTINEL — SYSTEMIC RISK ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Date: {graph_data['date']}",
        f"Market Regime: {graph_data['regime']} (VIX: {graph_data['vix']:.1f})",
        f"Network: {graph_data['metrics']['n_nodes']} nodes, {graph_data['metrics']['n_edges']} edges, "
        f"density {graph_data['metrics']['density']:.3f}",
        "",
        "--- SHOCK SCENARIO ---",
        f"Target: {summary['shocked_node']}",
        f"Magnitude: {summary['shock_magnitude']*100:.0f}%",
        f"Model: {summary['model']}",
        "",
        "--- RESULTS ---",
        f"Nodes Affected: {summary['n_affected']}",
        f"Nodes Defaulted: {summary['n_defaulted']}",
        f"Cascade Depth: {summary['cascade_depth']} waves",
        f"Total Systemic Stress: {summary['total_stress']:.2f}",
        f"Average Stress: {summary['avg_stress']*100:.2f}%",
        "",
        "--- TOP 10 AFFECTED ---",
    ]
    sectors = sector_dict or {}
    for item in summary["top_10_affected"]:
        ticker = item["ticker"]
        lines.append(f"  {ticker:6s} ({sectors.get(ticker, '?'):30s}) stress={item['stress']*100:.1f}%")

    lines.extend(["", "--- SECTOR BREAKDOWN ---"])
    sev_df = build_severity_df(shock_result, sectors)
    for _, row in sev_df.iterrows():
        lines.append(f"  {row['Sector']:30s} avg={row['Avg Stress %']:5.1f}%  hit={row['Nodes Hit']}/{row['Total']}")

    lines.extend(["", "--- AGENT MESSAGES ---"])
    for name, _icon, _css, text in agent_messages:
        clean = re.sub(r"<[^>]+>", "", text)
        lines.append(f"[{name}] {clean}")

    lines.extend([
        "",
        "=" * 60,
        "Generated by RiskSentinel — Microsoft AI Dev Days Hackathon 2026",
        "=" * 60,
    ])
    return "\n".join(lines)


def generate_report_markdown(
    *,
    graph_data: dict | None,
    shock_result,
    last_run_metrics: dict | None,
) -> str:
    if not graph_data or not shock_result:
        return "# RiskSentinel Brief\n\nNo simulation results available."

    summary = shock_result.summary()
    run_metrics = last_run_metrics or {}
    lines = [
        "# RiskSentinel Executive Brief",
        "",
        f"- **Date**: {graph_data['date']}",
        f"- **Regime**: {graph_data['regime']} (VIX {graph_data['vix']:.1f})",
        f"- **Scenario**: {summary['shocked_node']} shock {summary['shock_magnitude']*100:.0f}% with {summary['model']}",
        (
            f"- **Impact**: {summary['n_affected']} affected, {summary['n_defaulted']} defaulted, "
            f"{summary['cascade_depth']} waves, avg stress {summary['avg_stress']*100:.2f}%"
        ),
        (
            f"- **Runtime**: total {run_metrics.get('total_sec', 0.0):.1f}s, "
            f"local {run_metrics.get('local_sec', 0.0) if isinstance(run_metrics.get('local_sec'), float) else 0.0:.1f}s, "
            f"gpt {run_metrics.get('gpt_sec', 0.0) if isinstance(run_metrics.get('gpt_sec'), float) else 0.0:.1f}s"
        ),
        "",
        "## Top 5 Impacted Nodes",
    ]
    for item in summary["top_10_affected"][:5]:
        lines.append(f"- {item['ticker']}: {item['stress']*100:.1f}%")

    lines.extend(
        [
            "",
            "## Suggested Actions",
            "- Hedge concentrated sector exposures.",
            "- Reduce direct exposure to high-centrality nodes.",
            "- Monitor VIX and contagion breadth for escalation triggers.",
        ]
    )
    return "\n".join(lines)


def generate_action_pack_ceo_brief(
    *,
    graph_data: dict | None,
    shock_result,
    commander: dict | None,
    autonomous: dict | None,
    portfolio: dict | None,
) -> str:
    shock_summary = shock_result.summary() if shock_result else None
    return reporting.generate_action_pack_ceo_brief(
        graph_data=graph_data,
        shock_summary=shock_summary,
        commander=commander or {},
        autonomous=autonomous or {},
        portfolio=portfolio or {},
    )


def generate_action_pack_runbook(
    *,
    commander: dict | None,
    portfolio: dict | None,
) -> str:
    return reporting.generate_action_pack_runbook(
        commander=commander or {},
        portfolio=portfolio or {},
    )


def generate_action_pack_machine_json(
    *,
    graph_data: dict | None,
    commander: dict | None,
    autonomous_stress_test: dict | None,
    portfolio_copilot: dict | None,
    trace_summary: dict | None,
    policy_plan: list | None,
    executor_log: list | None,
    session_memory: list | None,
) -> str:
    payload = reporting.build_action_pack_payload(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        market_context=graph_data,
        commander=commander,
        autonomous_stress_test=autonomous_stress_test,
        portfolio_copilot=portfolio_copilot,
        trace_summary=trace_summary,
        policy_plan=policy_plan,
        executor_log=executor_log,
        session_memory=session_memory,
    )
    return reporting.generate_action_pack_machine_json(payload)


def generate_trace_bundle_json(
    *,
    last_run_metrics: dict | None,
    run_trace: dict | None,
    run_trace_history: list[dict],
    rag_last_docs: list | None,
    risk_profile: str,
    latest_policy_plan: list | None,
    latest_executor_log: list | None,
    session_decisions: list | None,
    commander_results: dict | None,
    autonomous_results: dict | None,
    portfolio_copilot: dict | None,
) -> str:
    quality_summary = summarize_quality(run_trace_history)
    judge_kpis = build_judge_kpis(run_trace_history)
    payload = {
        "last_run_metrics": last_run_metrics,
        "trace": run_trace,
        "history_size": len(run_trace_history),
        "quality_summary": quality_summary,
        "judge_kpis": judge_kpis,
        "rag_last_docs": rag_last_docs,
        "risk_profile": risk_profile,
        "policy_plan": latest_policy_plan,
        "executor_log": latest_executor_log,
        "session_decisions": (session_decisions or [])[-20:],
        "scenario_commander": commander_results,
        "autonomous_stress_test": autonomous_results,
        "portfolio_copilot": portfolio_copilot,
    }
    return json.dumps(reporting.json_safe(payload), indent=2)


def build_submission_bundle_bytes(
    *,
    report_text: str,
    brief_markdown: str,
    action_ceo_brief: str,
    action_runbook: str,
    action_machine_json: str,
    trace_json: str,
    run_trace_history: list[dict],
    scenario_eval_results: dict | None,
    rag_last_docs: list | None,
) -> bytes:
    quality = summarize_quality(run_trace_history)
    judge_kpis = build_judge_kpis(run_trace_history)
    quality_json = json.dumps(quality, indent=2)
    judge_json = json.dumps(judge_kpis, indent=2)
    scenario_eval_json = json.dumps(scenario_eval_results or {}, indent=2)
    rag_json = json.dumps(rag_last_docs or [], indent=2)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.txt", report_text)
        zf.writestr("executive_brief.md", brief_markdown)
        zf.writestr("explainability_trace.json", trace_json)
        zf.writestr("kpi_snapshot.json", quality_json)
        zf.writestr("judge_dashboard_kpis.json", judge_json)
        zf.writestr("evidence_rag_last_docs.json", rag_json)
        zf.writestr("scenario_pack_eval.json", scenario_eval_json)
        zf.writestr("action_pack_ceo_brief.md", action_ceo_brief)
        zf.writestr("action_pack_risk_runbook.md", action_runbook)
        zf.writestr("action_pack_machine.json", action_machine_json)
    return buf.getvalue()
