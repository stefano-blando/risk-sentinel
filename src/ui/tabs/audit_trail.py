"""Renderer for the Audit Trail tab."""

from __future__ import annotations


def render_tab(ctx: dict[str, object]) -> None:
    st = ctx["st"]
    pd = ctx["pd"]
    np = ctx["np"]
    ui_panels = ctx["ui_panels"]
    summarize_quality = ctx["summarize_quality"]
    build_judge_kpis = ctx["build_judge_kpis"]
    build_judge_run_rows = ctx["build_judge_run_rows"]
    SCENARIO_PACK = ctx["SCENARIO_PACK"]
    generate_report_text = ctx["generate_report_text"]
    generate_report_markdown = ctx["generate_report_markdown"]
    generate_trace_bundle_json = ctx["generate_trace_bundle_json"]
    build_submission_bundle_bytes = ctx["build_submission_bundle_bytes"]
    generate_action_pack_ceo_brief = ctx["generate_action_pack_ceo_brief"]
    generate_action_pack_runbook = ctx["generate_action_pack_runbook"]
    generate_action_pack_machine_json = ctx["generate_action_pack_machine_json"]
    selected_date = ctx["selected_date"]

    if not st.session_state.show_explainability:
        st.info("Enable `Show explainability panel` from `Ops` to view trace details.")
    elif st.session_state.run_trace:
        trace = st.session_state.run_trace
        st.markdown("### 🔍 Audit Trail")
        p = trace.get("policy", {})
        r = trace.get("result", {})
        t = trace.get("timings", {})

        xcols = st.columns(3)
        xcols[0].metric("Route", r.get("state", "n/a"))
        xcols[1].metric("Cache hit", "Yes" if p.get("cache_hit") else "No")
        xcols[2].metric("In scope", "Yes" if trace.get("in_scope") else "No")
        planned_steps = p.get("planned_steps") or st.session_state.latest_policy_plan
        exec_rows = p.get("executor_log") or st.session_state.latest_executor_log
        planner_status = "PASS" if planned_steps else "N/A"
        executor_status = "PASS" if exec_rows else "N/A"
        critic_approved = r.get("critic_approved")
        critic_rounds = r.get("critic_rounds")
        badges_html = " ".join(
            [
                ui_panels.stage_badge_html("Planner", planner_status),
                ui_panels.stage_badge_html("Executor", executor_status),
                ui_panels.critic_badge_html(critic_approved, critic_rounds),
            ]
        )
        st.markdown(badges_html, unsafe_allow_html=True)

        with st.expander("Decision policy", expanded=False):
            st.json(
                {
                    "scope_reason": trace.get("scope_reason"),
                    "complex_query": trace.get("complex_query"),
                    "router": p.get("router"),
                    "workflow": trace.get("workflow"),
                    "strategy": p.get("strategy"),
                    "should_run_gpt": p.get("should_run_gpt"),
                    "gpt_access_allowed": p.get("gpt_access_allowed"),
                    "gpt_access_reason": p.get("gpt_access_reason"),
                    "gpt_block_reason": p.get("gpt_block_reason", "none"),
                    "cache_mode": p.get("cache_mode", "n/a"),
                    "facts_mode": p.get("facts_mode", "none"),
                    "planned_steps_count": len(p.get("planned_steps", []) or []),
                    "session_memory_enabled": st.session_state.use_session_memory,
                    "critic_auto_repair": st.session_state.critic_auto_repair,
                    "evidence_gate_strict": st.session_state.evidence_gate_strict,
                    "engine": r.get("engine"),
                    "timings": t,
                }
            )

        gate_info = r.get("local_evidence_gate")
        cache_gate_info = p.get("cache_rejection_evidence_gate")
        if gate_info or cache_gate_info:
            with st.expander("Evidence gate checks", expanded=False):
                if gate_info:
                    st.json({"runtime_gate": gate_info})
                    if not gate_info.get("approved", True):
                        st.warning("Runtime evidence gate flagged issues before/with critic validation.")
                if cache_gate_info:
                    st.json({"cache_gate_rejection": cache_gate_info})
                    st.info("One cached answer was rejected by evidence gate and not reused.")

        with st.expander("Policy ↔ Executor Split", expanded=False):
            if planned_steps:
                st.markdown("**Policy Plan**")
                for idx, step_text in enumerate(planned_steps, start=1):
                    st.markdown(f"{idx}. {step_text}")
            if exec_rows:
                st.markdown("**Executor Timeline**")
                st.dataframe(pd.DataFrame(exec_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No executor timeline available yet.")

        with st.expander("Execution trace", expanded=False):
            events = trace.get("events", [])
            if events:
                st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
            else:
                st.caption("No trace events.")

        if p.get("facts_preview"):
            with st.expander("Deterministic facts injected into GPT", expanded=False):
                st.code(p["facts_preview"], language="text")

        if st.session_state.rag_last_docs:
            with st.expander("Evidence-RAG retrieval", expanded=False):
                rag_df = pd.DataFrame(st.session_state.rag_last_docs)
                show_cols = [c for c in ["reference_id", "source", "title", "score", "text"] if c in rag_df.columns]
                st.dataframe(rag_df[show_cols], use_container_width=True, hide_index=True)

        if st.session_state.session_decisions:
            with st.expander("Session decision memory", expanded=False):
                mem_df = pd.DataFrame(st.session_state.session_decisions[-20:])
                st.dataframe(mem_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trace available yet. Run a query first.")

    history = st.session_state.run_trace_history
    if history:
        states = [h.get("result", {}).get("state", "n/a") for h in history]
        hcols = st.columns(3)
        hcols[0].metric("Runs tracked", len(history))
        hcols[1].metric("GPT success", f"{sum(1 for h in history if h.get('result', {}).get('gpt_success'))}/{len(history)}")
        hcols[2].metric("Avg total", f"{np.mean([h.get('timings', {}).get('total_sec', 0.0) for h in history]):.1f}s")
        st.caption("Recent route states: " + ", ".join(pd.Series(states).value_counts().head(4).index.tolist()))
        st.caption(
            f"Session GPT calls: {st.session_state.gpt_calls_total_session} | "
            f"Policy throttles: {st.session_state.gpt_rate_limit_hits} | "
            f"Fail streak: {st.session_state.gpt_fail_streak}"
        )

        quality = summarize_quality(history)
        if quality:
            st.markdown("### ✅ Run Quality")
            qcols = st.columns(5)
            qcols[0].metric("Factual", f"{quality['factual_consistency_pct']:.1f}%" if quality["factual_consistency_pct"] is not None else "n/a")
            qcols[1].metric("Fallback rate", f"{quality['fallback_rate_pct']:.1f}%")
            qcols[2].metric("Cache hit", f"{quality['cache_hit_rate_pct']:.1f}%")
            qcols[3].metric("429 events", str(quality["rate_limit_events_total"]))
            qcols[4].metric("Avg uncertainty", f"{quality['avg_uncertainty']:.2f}")

        judge_kpis = build_judge_kpis(history)
        st.session_state.judge_kpis = judge_kpis
        if judge_kpis:
            st.markdown("### 🏁 Judge Dashboard")
            jcols = st.columns(5)
            jcols[0].metric("Critic pass-rate", f"{judge_kpis.get('critic_pass_rate_pct', 0.0):.1f}%")
            jcols[1].metric("Factual consistency", f"{judge_kpis.get('factual_consistency_pct', 0.0):.1f}%")
            jcols[2].metric("Latency p95", f"{judge_kpis.get('latency_p95_sec', 0.0):.2f}s")
            jcols[3].metric("Fallback rate", f"{judge_kpis.get('fallback_rate_pct', 0.0):.1f}%")
            gpt_runs = int(judge_kpis.get("gpt_runs", 0))
            gpt_ok = int(judge_kpis.get("gpt_success_runs", 0))
            jcols[4].metric("GPT success", f"{gpt_ok}/{gpt_runs}" if gpt_runs > 0 else "n/a")
            judge_rows_df = build_judge_run_rows(history, limit=20)
            if not judge_rows_df.empty:
                with st.expander("Judge run table (latest 20)", expanded=False):
                    st.dataframe(judge_rows_df, use_container_width=True, hide_index=True)

    if st.session_state.eval_results:
        st.markdown("### 🧪 Benchmark Results")
        er = st.session_state.eval_results
        st.caption(f"{er['n_ok']}/{er['n_queries']} success | avg latency {er['avg_latency_s']:.2f}s | total {er['total_time_s']:.2f}s")
        st.dataframe(pd.DataFrame(er["rows"]), use_container_width=True, hide_index=True)

    if st.session_state.scenario_eval_results:
        st.markdown("### 🧩 Scenario Pack Eval")
        se = st.session_state.scenario_eval_results
        st.caption(f"{se['n_pass']}/{se['n_scenarios']} PASS ({se['pass_rate_pct']:.1f}%)")
        st.dataframe(pd.DataFrame(se["rows"]), use_container_width=True, hide_index=True)

    if st.session_state.run_trace:
        current_query = st.session_state.run_trace.get("query", "")
        matched = next((s for s in SCENARIO_PACK if s["query"] == current_query), None)
        if matched:
            expected = matched["expected_route"]
            actual = st.session_state.run_trace.get("result", {}).get("state", "n/a")
            ok = expected in actual or (expected == "gpt" and actual.startswith("gpt_"))
            st.markdown("### 🎯 Scenario Check")
            st.caption(f"{matched['name']} | expected: {expected} | actual: {actual} | status: {'PASS' if ok else 'CHECK'}")

    if st.session_state.shock_result:
        sr = st.session_state.shock_result
        st.divider()
        report = generate_report_text()
        brief_md = generate_report_markdown()
        trace_json = generate_trace_bundle_json()
        submission_zip = build_submission_bundle_bytes()
        dcols = st.columns(3)
        dcols[0].download_button(
            "📥 Report (.txt)",
            report,
            file_name=f"risksentinel_report_{sr.shocked_node}_{st.session_state.graph_data['date']}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        dcols[1].download_button(
            "📄 Executive Brief (.md)",
            brief_md,
            file_name=f"risksentinel_brief_{sr.shocked_node}_{st.session_state.graph_data['date']}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        dcols[2].download_button(
            "🧾 Audit Trail (.json)",
            trace_json,
            file_name=f"risksentinel_trace_{sr.shocked_node}_{st.session_state.graph_data['date']}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "📦 Submission Bundle (.zip)",
            submission_zip,
            file_name=f"risksentinel_submission_bundle_{sr.shocked_node}_{st.session_state.graph_data['date']}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if (
        st.session_state.shock_result
        or st.session_state.commander_results
        or st.session_state.autonomous_results
        or st.session_state.portfolio_copilot
    ):
        st.markdown("### 🎬 Action Pack")
        action_ceo = generate_action_pack_ceo_brief()
        action_runbook = generate_action_pack_runbook()
        action_json = generate_action_pack_machine_json()
        action_date = (
            (st.session_state.graph_data or {}).get("date")
            or (st.session_state.commander_results or {}).get("date")
            or selected_date
        )
        acols = st.columns(3)
        acols[0].download_button(
            "🧭 CEO Brief (.md)",
            action_ceo,
            file_name=f"risksentinel_action_ceo_{action_date}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        acols[1].download_button(
            "📋 Risk Runbook (.md)",
            action_runbook,
            file_name=f"risksentinel_action_runbook_{action_date}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        acols[2].download_button(
            "🧩 Machine JSON (.json)",
            action_json,
            file_name=f"risksentinel_action_pack_{action_date}.json",
            mime="application/json",
            use_container_width=True,
        )
