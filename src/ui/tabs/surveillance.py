"""Renderer for the Surveillance tab."""

from __future__ import annotations


def render_tab(ctx: dict[str, object]) -> None:
    st = ctx["st"]
    pd = ctx["pd"]
    go = ctx["go"]
    MODEL_COLORS = ctx["MODEL_COLORS"]
    RISK_COLORS = ctx["RISK_COLORS"]
    PALETTE = ctx["PALETTE"]
    ui_panels = ctx["ui_panels"]
    build_systemic_risk_gauge_figure = ctx["build_systemic_risk_gauge_figure"]
    build_sector_impact_bar_figure = ctx["build_sector_impact_bar_figure"]
    build_stress_tier_donut_figure = ctx["build_stress_tier_donut_figure"]
    build_wave_trend_figure = ctx["build_wave_trend_figure"]
    build_timeline_figure = ctx["build_timeline_figure"]
    build_severity_df = ctx["build_severity_df"]

    st.markdown("### 🛰️ Systemic Surveillance")
    if st.session_state.shock_result and st.session_state.graph_data:
        sr = st.session_state.shock_result
        gd = st.session_state.graph_data
        summary = sr.summary()
        total_nodes = int(gd["metrics"]["n_nodes"])

        gauge_fig, risk_index, risk_label = build_systemic_risk_gauge_figure(sr, total_nodes)
        affected_pct = (summary["n_affected"] / max(1, total_nodes)) * 100.0

        kcols = st.columns(6)
        kcols[0].metric("Affected", f"{summary['n_affected']}/{total_nodes}", f"{affected_pct:.1f}%")
        kcols[1].metric("Defaulted", summary["n_defaulted"])
        kcols[2].metric("Waves", summary["cascade_depth"])
        kcols[3].metric("Total Stress", f"{summary['total_stress']:.2f}")
        kcols[4].metric("Avg Stress", f"{summary['avg_stress']*100:.1f}%")
        kcols[5].metric("Risk Index", f"{risk_index:.0f}", risk_label)

        dcol1, dcol2 = st.columns([2, 1])
        with dcol1:
            st.plotly_chart(build_sector_impact_bar_figure(sr), use_container_width=True, config={"displayModeBar": False})
        with dcol2:
            st.plotly_chart(build_stress_tier_donut_figure(sr), use_container_width=True, config={"displayModeBar": False})

        dcol3, dcol4 = st.columns([1, 2])
        with dcol3:
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
        with dcol4:
            st.plotly_chart(build_wave_trend_figure(sr), use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Run a shock scenario to populate the live dashboard.")

    st.divider()
    st.markdown("### ⚖️ Model Comparison")
    if st.session_state.comparison:
        comp = st.session_state.comparison
        comp_rows = []
        for model_name, res in comp.items():
            s = res.summary()
            comp_rows.append(
                {
                    "Model": model_name.replace("_", " ").title(),
                    "Affected": s["n_affected"],
                    "Defaulted": s["n_defaulted"],
                    "Waves": s["cascade_depth"],
                    "Avg Stress %": round(s["avg_stress"] * 100, 1),
                    "Total Stress": round(s["total_stress"], 1),
                }
            )
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(
            comp_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Avg Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            },
        )

        fig_comp = go.Figure()
        for _, row in comp_df.iterrows():
            fig_comp.add_trace(
                go.Bar(
                    name=row["Model"],
                    x=["Affected", "Defaulted", "Waves"],
                    y=[row["Affected"], row["Defaulted"], row["Waves"]],
                    marker_color=MODEL_COLORS.get(row["Model"], RISK_COLORS["none"]),
                    text=[row["Affected"], row["Defaulted"], row["Waves"]],
                    textposition="auto",
                )
            )
        fig_comp.update_layout(
            barmode="group",
            height=260,
            margin=dict(l=40, r=20, t=20, b=30),
            plot_bgcolor=PALETTE["bg_main"],
            paper_bgcolor=PALETTE["bg_main"],
            font=dict(color=PALETTE["text_primary"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=PALETTE["text_muted"], size=11)),
            xaxis=dict(color=PALETTE["text_muted"]),
            yaxis=dict(color=PALETTE["text_muted"], showgrid=True, gridcolor=PALETTE["surface_1"]),
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Run `Compare All 3 Models` from the sidebar to populate this section.")

    st.markdown("### 📈 Network Health Timeline")
    timeline_fig = build_timeline_figure()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

    if st.session_state.commander_results:
        st.markdown("### 🧭 Scenario Commander")
        cmd = st.session_state.commander_results
        st.caption(
            f"Date {cmd.get('date')} | Regime {cmd.get('regime')} (VIX {cmd.get('vix', 0.0):.1f}) | "
            f"Shock {cmd.get('shock_pct')}% via {cmd.get('model')}"
        )
        cmd_df = pd.DataFrame(cmd.get("rows", []))
        if not cmd_df.empty:
            show_cols = [c for c in ["rank", "ticker", "sector", "risk_score", "n_affected", "cascade_depth", "avg_stress_pct"] if c in cmd_df.columns]
            st.dataframe(
                cmd_df[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                    "avg_stress_pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                },
            )

    if st.session_state.autonomous_results:
        st.markdown("### 🛰️ Autonomous Stress Test")
        auto = st.session_state.autonomous_results
        st.caption(
            f"Seed tickers: {len(auto.get('seed_tickers', []))} | "
            f"Shock grid: {auto.get('shock_grid')} | Model: {auto.get('model')}"
        )
        auto_df = pd.DataFrame(auto.get("rows", []))
        if not auto_df.empty:
            show_cols = [c for c in ["ticker", "sector", "shock_pct", "risk_score", "n_affected", "cascade_depth", "avg_stress_pct"] if c in auto_df.columns]
            st.dataframe(
                auto_df[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                    "avg_stress_pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                },
            )

    if st.session_state.portfolio_copilot:
        st.markdown("### 📦 Portfolio Co-Pilot")
        cop = st.session_state.portfolio_copilot
        if not cop.get("ok"):
            errs = cop.get("errors", [])
            st.warning("Portfolio input not valid." + (" " + " | ".join(errs[:4]) if errs else ""))
        else:
            st.caption(
                f"Expected stress: {cop.get('expected_stress_pct', 0.0):.1f}% "
                f"→ {cop.get('expected_stress_pct_after_hedge', 0.0):.1f}% after hedge "
                f"(avoided ~{cop.get('estimated_loss_avoided_pct', 0.0):.1f}%)."
            )
            formula_md = ui_panels.business_kpi_formula_markdown(cop.get("kpi"))
            if formula_md:
                st.markdown(formula_md)
            pos_df = pd.DataFrame(cop.get("positions", []))
            if not pos_df.empty:
                show_cols = [
                    c for c in
                    ["ticker", "sector", "weight_norm_pct", "risk_score", "weighted_risk", "avg_stress_pct", "n_affected", "cascade_depth"]
                    if c in pos_df.columns
                ]
                st.dataframe(
                    pos_df[show_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "weight_norm_pct": st.column_config.ProgressColumn(min_value=-100, max_value=100, format="%.2f%%"),
                        "risk_score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                        "avg_stress_pct": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                    },
                )
            if cop.get("actions"):
                st.markdown("**Suggested Hedge Actions**")
                for action in cop["actions"][:6]:
                    st.markdown(f"- {action}")

    if st.session_state.shock_result:
        sr = st.session_state.shock_result
        st.markdown("### 📊 Sector Impact")
        sev_df = build_severity_df(sr)
        st.dataframe(
            sev_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Avg Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")},
        )

        st.markdown("### 🎯 Most Vulnerable")
        affected = sr.affected_nodes[:10]
        if affected:
            df = pd.DataFrame(affected, columns=["Ticker", "Stress"])
            df["Sector"] = df["Ticker"].map(st.session_state.sector_dict)
            df["Stress %"] = (df["Stress"] * 100).round(1)
            st.dataframe(
                df[["Ticker", "Sector", "Stress %"]],
                use_container_width=True,
                hide_index=True,
                column_config={"Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")},
            )
