"""Renderer for the Stress Lab tab."""

from __future__ import annotations


def render_tab(ctx: dict[str, object]) -> None:
    st = ctx["st"]
    build_animated_figure = ctx["build_animated_figure"]
    build_graph_figure = ctx["build_graph_figure"]
    agent_message = ctx["agent_message"]
    build_compare_rows_df = ctx["build_compare_rows_df"]
    MAX_COMPARE_TICKERS = ctx["MAX_COMPARE_TICKERS"]

    graph_col, info_col = st.columns([3, 2])

    with graph_col:
        st.markdown("### 🌐 Correlation Network")
        if st.session_state.graph_data:
            G = st.session_state.graph_data["G"]
            pos = st.session_state.pos
            sr = st.session_state.shock_result

            if sr:
                blast_view = st.toggle(
                    "🎯 Blast radius only",
                    value=False,
                    key="blast_radius_simulate",
                    help="Show only affected subgraph",
                )
                fig = build_animated_figure(G, pos, sr, blast_radius_only=blast_view)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
                st.markdown(
                    "⚪ Shocked &nbsp; 🔴 Critical &nbsp; 🟠 High &nbsp; "
                    "🟡 Moderate &nbsp; 🔵 Low &nbsp; ⚫ Unaffected &emsp; | &emsp; "
                    "Use **▶ Play** or drag the **slider** below the graph"
                )
            else:
                fig = build_graph_figure(G, pos)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
        else:
            st.info("Build a network from the sidebar to start simulation.")

    with info_col:
        st.markdown("### 🤖 Agent Analysis")
        run_metrics = st.session_state.last_run_metrics
        if run_metrics:
            badge_map = {
                "gpt_ok": "GPT OK",
                "gpt_cached": "GPT Cached",
                "gpt_retry_ok": "GPT Retry OK",
                "gpt_fallback_ok": "GPT Fallback OK",
                "gpt_failed": "GPT Failed",
                "gpt_failed_local_fallback": "GPT Failed (Local Fallback)",
                "local_fast_mode": "Local Fast Mode",
                "local_only": "Local Only",
                "parse_failed": "Parse Failed",
                "out_of_scope": "Out Of Scope",
                "gpt_policy_block_local": "GPT Blocked (Local Only)",
                "gpt_policy_block_parse_failed": "GPT Blocked + Parse Failed",
                "cancelled": "Cancelled",
            }
            st.caption(f"Last run: {badge_map.get(run_metrics.get('state', ''), run_metrics.get('state', 'n/a'))}")
            tcols = st.columns(5)
            tcols[0].metric("Total", f"{run_metrics.get('total_sec', 0.0):.1f}s")
            local_val = run_metrics.get("local_sec")
            tcols[1].metric("Local", f"{local_val:.1f}s" if isinstance(local_val, float) else "n/a")
            gpt_val = run_metrics.get("gpt_sec")
            tcols[2].metric("GPT", f"{gpt_val:.1f}s" if isinstance(gpt_val, float) else "n/a")
            tcols[3].metric("Engine", run_metrics.get("engine", "n/a"))
            c_rounds = run_metrics.get("critic_rounds")
            tcols[4].metric("Critic rounds", str(c_rounds) if c_rounds else "n/a")

        if st.session_state.agent_messages:
            for name, icon, css, text in st.session_state.agent_messages:
                agent_message(name, icon, css, text)
        else:
            st.info("Type a question below, use a **Crisis Preset**, or click **Build** → **Shock**.")

        if st.session_state.compare_rows_local:
            st.markdown("### 🧮 Deterministic Compare")
            compare_df = build_compare_rows_df(st.session_state.compare_rows_local)
            if not compare_df.empty:
                st.dataframe(
                    compare_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "avg_stress_pct": st.column_config.ProgressColumn(
                            "avg_stress_pct", min_value=0, max_value=100, format="%.2f%%"
                        ),
                    },
                )
                meta = st.session_state.compare_meta or {}
                req = len(meta.get("requested_tickers", []))
                ev = len(meta.get("evaluated_tickers", []))
                if req > ev:
                    st.caption(
                        f"Compared {ev}/{req} tickers (limit {meta.get('max_tickers', MAX_COMPARE_TICKERS)} per run)."
                    )
