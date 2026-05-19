"""Streamlit-safe helpers for sidebar agentic actions."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def run_sidebar_agentic_actions(
    *,
    st_module,
    session_state,
    selected_date: str,
    threshold: float,
    shock_pct: int,
    shock_model: str,
    sector_dict_ctx: dict[str, str],
    tickers_ctx: list[str],
    risk_profile_ctx: str,
    auto_portfolio_n_ctx: int,
    portfolio_text_ctx: str,
    commander_btn: bool,
    autonomous_btn: bool,
    auto_portfolio_btn: bool,
    portfolio_btn: bool,
    full_demo_btn: bool,
    default_commander_top_n: int,
    autonomous_shock_grid,
    default_autonomous_seeds: int,
    cache_key_fn,
    run_agentic_operation_fn,
    run_scenario_commander_fn,
    run_autonomous_stress_test_fn,
    build_auto_portfolio_from_network_fn,
    run_portfolio_copilot_fn,
    do_build_network_fn,
    agentic_ops_module,
) -> None:
    if commander_btn and sector_dict_ctx:
        cache_key = cache_key_fn(
            "scenario_commander",
            date=selected_date,
            threshold=round(float(threshold), 3),
            shock_pct=int(shock_pct),
            model=shock_model,
            top_n=default_commander_top_n,
        )
        with st_module.spinner("Running Scenario Commander..."):
            result, from_cache = run_agentic_operation_fn(
                op_name="Scenario Commander",
                cache_key=cache_key,
                fn=lambda: run_scenario_commander_fn(
                    date_str=selected_date,
                    threshold=threshold,
                    shock_pct=shock_pct,
                    model=shock_model,
                    top_n=default_commander_top_n,
                    sector_dict=sector_dict_ctx,
                ),
            )
        if not result.get("ok", True):
            session_state.last_agentic_action = (
                f"Scenario Commander failed: {result.get('error', 'unknown error')}."
            )
        else:
            session_state.commander_results = result
            top = result.get("top_pick")
            if top:
                cache_suffix = " (cached)" if from_cache else ""
                session_state.agent_messages.append(
                    (
                        "Sentinel",
                        "🛡️",
                        "agent-sentinel",
                        f"Scenario Commander completed{cache_suffix}. Top systemic seed: "
                        f"<b>{top['ticker']}</b> (score {top['risk_score']:.1f}, "
                        f"depth {top['cascade_depth']}).",
                    )
                )
                session_state.last_agentic_action = (
                    f"Scenario Commander done{cache_suffix}: top={top['ticker']} "
                    f"score={top['risk_score']:.1f}. Open Surveillance tab for full ranking."
                )

    if autonomous_btn and sector_dict_ctx:
        cache_key = cache_key_fn(
            "autonomous_stress_test",
            date=selected_date,
            threshold=round(float(threshold), 3),
            model=shock_model,
            shock_grid=autonomous_shock_grid,
            max_seeds=default_autonomous_seeds,
        )
        with st_module.spinner("Running Autonomous Stress Test..."):
            result, from_cache = run_agentic_operation_fn(
                op_name="Autonomous Stress Test",
                cache_key=cache_key,
                fn=lambda: run_autonomous_stress_test_fn(
                    date_str=selected_date,
                    threshold=threshold,
                    model=shock_model,
                    sector_dict=sector_dict_ctx,
                ),
            )
        if not result.get("ok", True):
            session_state.last_agentic_action = (
                f"Auto Stress failed: {result.get('error', 'unknown error')}."
            )
        else:
            session_state.autonomous_results = result
            rows = result.get("rows") or []
            if rows:
                lead = rows[0]
                cache_suffix = " (cached)" if from_cache else ""
                session_state.agent_messages.append(
                    (
                        "Sentinel",
                        "🛡️",
                        "agent-sentinel",
                        f"Autonomous Stress Test completed{cache_suffix}. Lead fragility: "
                        f"<b>{lead['ticker']}</b> @ {lead['shock_pct']}% "
                        f"(score {lead['risk_score']:.1f}).",
                    )
                )
                session_state.last_agentic_action = (
                    f"Auto Stress done{cache_suffix}: lead={lead['ticker']} "
                    f"shock={lead['shock_pct']}% score={lead['risk_score']:.1f}. "
                    "Open Surveillance tab for table."
                )

    if auto_portfolio_btn and sector_dict_ctx:
        cache_key = cache_key_fn(
            "auto_portfolio",
            date=selected_date,
            threshold=round(float(threshold), 3),
            n_positions=int(session_state.auto_portfolio_n),
        )
        with st_module.spinner("Generating portfolio from current network..."):
            result, from_cache = run_agentic_operation_fn(
                op_name="Auto Portfolio",
                cache_key=cache_key,
                fn=lambda: build_auto_portfolio_from_network_fn(
                    date_str=selected_date,
                    threshold=threshold,
                    n_positions=auto_portfolio_n_ctx,
                    sector_dict=sector_dict_ctx,
                ),
            )
        if result.get("ok"):
            session_state.portfolio_text = result.get("portfolio_text", "")
            top = (result.get("rows") or [{}])[0]
            cache_suffix = " (cached)" if from_cache else ""
            session_state.last_agentic_action = (
                f"Auto-portfolio ready{cache_suffix} ({len(result.get('rows', []))} positions) "
                f"for {result.get('date')}, regime {result.get('regime')}."
            )
            session_state.agent_messages.append(
                (
                    "Architect",
                    "🔧",
                    "agent-architect",
                    "Auto portfolio created from network topology "
                    f"(method: {result.get('method')}{cache_suffix}). "
                    f"Top ticker: <b>{top.get('ticker', 'n/a')}</b>.",
                )
            )
        else:
            session_state.last_agentic_action = (
                f"Auto-portfolio failed: {result.get('error', 'unknown error')}."
            )

    if portfolio_btn and sector_dict_ctx and tickers_ctx:
        portfolio_hash = hashlib.sha1(portfolio_text_ctx.strip().encode("utf-8")).hexdigest()
        cache_key = cache_key_fn(
            "portfolio_copilot",
            portfolio_hash=portfolio_hash,
            date=selected_date,
            threshold=round(float(threshold), 3),
            model=shock_model,
            shock_pct=int(shock_pct),
            risk_profile=session_state.risk_profile,
        )
        with st_module.spinner("Running Portfolio Co-Pilot..."):
            result, from_cache = run_agentic_operation_fn(
                op_name="Portfolio Co-Pilot",
                cache_key=cache_key,
                fn=lambda: run_portfolio_copilot_fn(
                    portfolio_text=portfolio_text_ctx,
                    date_str=selected_date,
                    threshold=threshold,
                    model=shock_model,
                    stress_shock_pct=shock_pct,
                    risk_profile=risk_profile_ctx,
                    tickers=tickers_ctx,
                    sector_dict=sector_dict_ctx,
                ),
            )
            session_state.portfolio_copilot = result
        if result.get("ok"):
            cache_suffix = " (cached)" if from_cache else ""
            session_state.agent_messages.append(
                (
                    "Advisor",
                    "📋",
                    "agent-advisor",
                    f"Portfolio Co-Pilot ready{cache_suffix}. Expected stress: "
                    f"<b>{result.get('expected_stress_pct', 0.0):.1f}%</b> "
                    f"→ after hedges <b>{result.get('expected_stress_pct_after_hedge', 0.0):.1f}%</b>.",
                )
            )
            session_state.last_agentic_action = (
                f"Portfolio Co-Pilot done{cache_suffix}: stress "
                f"{result.get('expected_stress_pct', 0.0):.1f}% "
                f"-> {result.get('expected_stress_pct_after_hedge', 0.0):.1f}%."
            )
        elif result.get("timeout"):
            session_state.last_agentic_action = (
                f"Portfolio Co-Pilot timeout: {result.get('error', 'try again')}."
            )
        elif result.get("error"):
            session_state.last_agentic_action = f"Portfolio Co-Pilot failed: {result.get('error')}."
        else:
            errs = (result.get("errors") or [])[:3]
            session_state.last_agentic_action = (
                "Portfolio Co-Pilot input invalid. "
                + (" | ".join(errs) if errs else "Check ticker,weight format.")
            )

    if full_demo_btn and sector_dict_ctx and tickers_ctx:
        run = agentic_ops_module.build_full_demo_steps(now_utc=datetime.now(timezone.utc).isoformat())
        failed = False
        working_portfolio_text = portfolio_text_ctx
        with st_module.spinner("Running Full Agentic Demo..."):
            try:
                do_build_network_fn(selected_date, threshold)
                agentic_ops_module.append_demo_step(
                    run,
                    "Build network",
                    "ok",
                    f"date={selected_date} threshold={threshold:.2f}",
                )
            except Exception as exc:
                failed = True
                agentic_ops_module.append_demo_step(
                    run,
                    "Build network",
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                )

            if not failed:
                cache_key = cache_key_fn(
                    "scenario_commander",
                    date=selected_date,
                    threshold=round(float(threshold), 3),
                    shock_pct=int(shock_pct),
                    model=shock_model,
                    top_n=default_commander_top_n,
                )
                result, from_cache = run_agentic_operation_fn(
                    op_name="Scenario Commander",
                    cache_key=cache_key,
                    fn=lambda: run_scenario_commander_fn(
                        date_str=selected_date,
                        threshold=threshold,
                        shock_pct=shock_pct,
                        model=shock_model,
                        top_n=default_commander_top_n,
                        sector_dict=sector_dict_ctx,
                    ),
                )
                if result.get("ok", True):
                    session_state.commander_results = result
                    top = (result.get("top_pick") or {}).get("ticker", "n/a")
                    agentic_ops_module.append_demo_step(
                        run,
                        "Scenario Commander",
                        "ok",
                        f"top={top} cached={from_cache}",
                    )
                else:
                    failed = True
                    agentic_ops_module.append_demo_step(
                        run,
                        "Scenario Commander",
                        "failed",
                        result.get("error", "unknown error"),
                    )

            if not failed:
                cache_key = cache_key_fn(
                    "autonomous_stress_test",
                    date=selected_date,
                    threshold=round(float(threshold), 3),
                    model=shock_model,
                    shock_grid=autonomous_shock_grid,
                    max_seeds=default_autonomous_seeds,
                )
                result, from_cache = run_agentic_operation_fn(
                    op_name="Autonomous Stress Test",
                    cache_key=cache_key,
                    fn=lambda: run_autonomous_stress_test_fn(
                        date_str=selected_date,
                        threshold=threshold,
                        model=shock_model,
                        sector_dict=sector_dict_ctx,
                    ),
                )
                if result.get("ok", True):
                    session_state.autonomous_results = result
                    lead = ((result.get("rows") or [{}])[0]).get("ticker", "n/a")
                    agentic_ops_module.append_demo_step(
                        run,
                        "Autonomous Stress Test",
                        "ok",
                        f"lead={lead} cached={from_cache}",
                    )
                else:
                    failed = True
                    agentic_ops_module.append_demo_step(
                        run,
                        "Autonomous Stress Test",
                        "failed",
                        result.get("error", "unknown error"),
                    )

            if not failed and not working_portfolio_text.strip():
                cache_key = cache_key_fn(
                    "auto_portfolio",
                    date=selected_date,
                    threshold=round(float(threshold), 3),
                    n_positions=int(session_state.auto_portfolio_n),
                )
                result, from_cache = run_agentic_operation_fn(
                    op_name="Auto Portfolio",
                    cache_key=cache_key,
                    fn=lambda: build_auto_portfolio_from_network_fn(
                        date_str=selected_date,
                        threshold=threshold,
                        n_positions=auto_portfolio_n_ctx,
                        sector_dict=sector_dict_ctx,
                    ),
                )
                if result.get("ok"):
                    working_portfolio_text = str(result.get("portfolio_text", "") or "")
                    session_state.portfolio_text = working_portfolio_text
                    agentic_ops_module.append_demo_step(
                        run,
                        "Auto Portfolio",
                        "ok",
                        f"positions={len(result.get('rows', []))} cached={from_cache}",
                    )
                else:
                    failed = True
                    agentic_ops_module.append_demo_step(
                        run,
                        "Auto Portfolio",
                        "failed",
                        result.get("error", "unknown error"),
                    )

            if not failed:
                portfolio_hash = hashlib.sha1(
                    working_portfolio_text.strip().encode("utf-8")
                ).hexdigest()
                cache_key = cache_key_fn(
                    "portfolio_copilot",
                    portfolio_hash=portfolio_hash,
                    date=selected_date,
                    threshold=round(float(threshold), 3),
                    model=shock_model,
                    shock_pct=int(shock_pct),
                    risk_profile=session_state.risk_profile,
                )
                result, from_cache = run_agentic_operation_fn(
                    op_name="Portfolio Co-Pilot",
                    cache_key=cache_key,
                    fn=lambda: run_portfolio_copilot_fn(
                        portfolio_text=working_portfolio_text,
                        date_str=selected_date,
                        threshold=threshold,
                        model=shock_model,
                        stress_shock_pct=shock_pct,
                        risk_profile=risk_profile_ctx,
                        tickers=tickers_ctx,
                        sector_dict=sector_dict_ctx,
                    ),
                )
                session_state.portfolio_copilot = result
                if result.get("ok"):
                    agentic_ops_module.append_demo_step(
                        run,
                        "Portfolio Co-Pilot",
                        "ok",
                        (
                            f"stress={result.get('expected_stress_pct', 0.0):.1f}% "
                            f"avoided={result.get('estimated_loss_avoided_pct', 0.0):.1f}% "
                            f"cached={from_cache}"
                        ),
                    )
                else:
                    failed = True
                    detail = (
                        result.get("error")
                        or " | ".join((result.get("errors") or [])[:2])
                        or "invalid input"
                    )
                    agentic_ops_module.append_demo_step(run, "Portfolio Co-Pilot", "failed", detail)

        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if failed:
            has_ok = any(step.get("status") == "ok" for step in run.get("steps", []))
            run["status"] = "partial" if has_ok else "failed"
        else:
            run["status"] = "completed"
        session_state.full_demo_last_run = run
        if failed:
            failed_steps = [
                step["step"] for step in run.get("steps", []) if step.get("status") == "failed"
            ]
            session_state.last_agentic_action = (
                "Full Agentic Demo partial/failed. "
                + (
                    f"Failed step(s): {', '.join(failed_steps)}."
                    if failed_steps
                    else "Check logs."
                )
            )
        else:
            session_state.last_agentic_action = (
                "Full Agentic Demo completed (Build + Commander + Auto Stress + Co-Pilot)."
            )
            session_state.agent_messages.append(
                (
                    "Sentinel",
                    "🛡️",
                    "agent-sentinel",
                    "Full Agentic Demo completed. Review Surveillance and Audit Trail for "
                    "ranked vulnerabilities and evidence.",
                )
            )
