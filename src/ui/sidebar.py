"""Sidebar renderer for the Streamlit app shell."""

from __future__ import annotations


def render_sidebar(ctx: dict[str, object]) -> dict[str, object]:
    st = ctx["st"]
    pd = ctx["pd"]
    data_loader = ctx["data_loader"]
    DEMO_QUERIES = ctx["DEMO_QUERIES"]
    SCENARIO_PACK = ctx["SCENARIO_PACK"]
    CRISIS_PRESETS = ctx["CRISIS_PRESETS"]
    PORTFOLIO_SAMPLE = ctx["PORTFOLIO_SAMPLE"]

    preset_triggered = None

    with st.sidebar:
        st.markdown("## 🛡️ RiskSentinel")
        st.caption("Agentic Systemic Risk Simulator")
        if data_loader.is_synthetic_mode():
            st.info("Synthetic demo dataset active (cloud-safe fallback).")
        st.divider()

        st.markdown("### ⚡ Quick Actions")
        st.session_state.demo_mode = st.toggle(
            "Guided demo flow",
            value=st.session_state.demo_mode,
            help="Shows ready-to-run demo stories and one-click prompts for pitch sessions.",
        )
        if st.session_state.demo_mode:
            st.session_state.demo_story = st.selectbox(
                "Demo story",
                options=list(DEMO_QUERIES.keys()),
                index=list(DEMO_QUERIES.keys()).index(st.session_state.demo_story)
                if st.session_state.demo_story in DEMO_QUERIES else 0,
            )
            st.caption(DEMO_QUERIES[st.session_state.demo_story])
            if st.button("▶ Run Demo Query", use_container_width=True):
                st.session_state.pending_chat_query = DEMO_QUERIES[st.session_state.demo_story]
                st.rerun()

        st.markdown("### 🧩 Scenario Pack")
        scenario_names = [scenario["name"] for scenario in SCENARIO_PACK]
        st.session_state.scenario_pack_choice = st.selectbox(
            "Judge scenario",
            options=scenario_names,
            index=scenario_names.index(st.session_state.scenario_pack_choice)
            if st.session_state.scenario_pack_choice in scenario_names else 0,
        )
        selected_scenario = next(
            scenario for scenario in SCENARIO_PACK if scenario["name"] == st.session_state.scenario_pack_choice
        )
        st.caption(selected_scenario["query"])
        st.caption(f"Expected route: {selected_scenario['expected_route']}")
        if st.button("▶ Run Scenario", use_container_width=True):
            st.session_state.pending_chat_query = selected_scenario["query"]
            st.rerun()

        st.divider()

        st.markdown("### ⚡ Crisis Presets")
        preset_cols = st.columns(2)
        for idx, (name, params) in enumerate(CRISIS_PRESETS.items()):
            col = preset_cols[idx % 2]
            if col.button(name, key=f"preset_{name}", use_container_width=True):
                preset_triggered = {"name": name, "params": params}

        st.divider()

        st.markdown("### 📅 Network Date")
        available_dates = data_loader.get_available_dates()
        date_strings = [str(dt.date()) for dt in available_dates]
        init_date = st.session_state.sel_date or date_strings[-1]
        if init_date not in date_strings:
            init_date = min(date_strings, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(init_date)))
        selected_date = st.select_slider("Date", options=date_strings, value=init_date, label_visibility="collapsed")

        st.divider()
        st.markdown("### 💥 Shock Scenario")
        shocked_ticker = st.selectbox(
            "Target stock",
            options=st.session_state.tickers,
            index=st.session_state.tickers.index(st.session_state.sel_ticker)
            if st.session_state.sel_ticker in st.session_state.tickers else 0,
        )
        shock_pct = st.slider("Shock %", 10, 100, st.session_state.sel_shock, 10)
        model_options = ["debtrank", "linear_threshold", "cascade_removal"]
        shock_model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.sel_model)
            if st.session_state.sel_model in model_options
            else 0,
        )
        threshold = st.slider(
            "Corr. threshold",
            0.2,
            0.8,
            st.session_state.sel_threshold,
            0.05,
            help="Higher = sparser = more realistic contagion",
        )

        st.divider()
        col1, col2 = st.columns(2)
        build_btn = col1.button("🔨 Build", use_container_width=True)
        shock_btn = col2.button("💥 Shock", use_container_width=True, type="primary")
        compare_btn = st.button("⚖️ Compare All 3 Models", use_container_width=True)

        st.markdown("### 🤖 Agentic Ops")
        a1, a2 = st.columns(2)
        commander_btn = a1.button("🧭 Scenario Commander", use_container_width=True)
        autonomous_btn = a2.button("🛰️ Auto Stress Test", use_container_width=True)
        st.session_state.portfolio_text = st.text_area(
            "Portfolio (ticker,weight per line)",
            value=st.session_state.portfolio_text,
            height=120,
            placeholder=PORTFOLIO_SAMPLE,
            help="Editable input. Format: TICKER,weight (es. JPM,0.25).",
        )
        pcol1, pcol2 = st.columns(2)
        if pcol1.button("Load Sample Portfolio", use_container_width=True):
            st.session_state.portfolio_text = PORTFOLIO_SAMPLE
            st.session_state.last_agentic_action = "Sample portfolio loaded. Edit freely or run Co-Pilot."
            st.rerun()
        st.session_state.auto_portfolio_n = pcol2.selectbox(
            "Auto N",
            options=[3, 4, 5, 6, 8, 10],
            index=[3, 4, 5, 6, 8, 10].index(st.session_state.auto_portfolio_n)
            if st.session_state.auto_portfolio_n in {3, 4, 5, 6, 8, 10}
            else 2,
            help="Number of positions for auto-generated portfolio.",
        )
        auto_portfolio_btn = st.button("✨ Auto-generate from current network", use_container_width=True)
        portfolio_btn = st.button("📦 Portfolio Co-Pilot", use_container_width=True)
        full_demo_btn = st.button("🎬 Run Full Agentic Demo", use_container_width=True)

        st.caption("First `Build` loads correlation data into memory (~1GB on disk, ~1.4GB RAM process peak).")
        st.caption("Advanced GPT controls, diagnostics, and audit tools are in the `Ops` tab.")

    return {
        "preset_triggered": preset_triggered,
        "selected_date": selected_date,
        "shocked_ticker": shocked_ticker,
        "shock_pct": shock_pct,
        "shock_model": shock_model,
        "threshold": threshold,
        "build_btn": build_btn,
        "shock_btn": shock_btn,
        "compare_btn": compare_btn,
        "commander_btn": commander_btn,
        "autonomous_btn": autonomous_btn,
        "auto_portfolio_btn": auto_portfolio_btn,
        "portfolio_btn": portfolio_btn,
        "full_demo_btn": full_demo_btn,
    }
