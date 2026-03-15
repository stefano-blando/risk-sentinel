"""Renderer for the Outlook tab."""

from __future__ import annotations


def render_tab(ctx: dict[str, object]) -> None:
    st = ctx["st"]
    pd = ctx["pd"]
    np = ctx["np"]
    html = ctx["html"]
    json = ctx["json"]
    data_loader = ctx["data_loader"]
    forecast_nearest_leq = ctx["forecast_nearest_leq"]
    _get_forecast_date_bounds = ctx["_get_forecast_date_bounds"]
    _run_live_outlook_cached = ctx["_run_live_outlook_cached"]
    load_forecast_report = ctx["load_forecast_report"]
    _get_forecast_report_path = ctx["_get_forecast_report_path"]
    _summary_rows_from_forecast = ctx["_summary_rows_from_forecast"]
    _compute_outlook_snapshot = ctx["_compute_outlook_snapshot"]
    _build_change_rows = ctx["_build_change_rows"]
    _regime_pill_html = ctx["_regime_pill_html"]
    _build_regime_transition_copy = ctx["_build_regime_transition_copy"]
    _forecast_confidence_copy = ctx["_forecast_confidence_copy"]
    _format_outlook_metric_label = ctx["_format_outlook_metric_label"]
    build_outlook_compact_figure = ctx["build_outlook_compact_figure"]
    build_outlook_spread_figure = ctx["build_outlook_spread_figure"]
    _build_outlook_checkpoint_rows = ctx["_build_outlook_checkpoint_rows"]
    build_outlook_animation_figure = ctx["build_outlook_animation_figure"]
    build_outlook_timeseries_figure = ctx["build_outlook_timeseries_figure"]
    _build_narrative_lines = ctx["_build_narrative_lines"]
    _top_systemic_rows = ctx["_top_systemic_rows"]
    _build_action_rows = ctx["_build_action_rows"]
    _build_why_this_matters_rows = ctx["_build_why_this_matters_rows"]
    _build_watchlist_rows = ctx["_build_watchlist_rows"]
    OUTLOOK_SCENARIOS = ctx["OUTLOOK_SCENARIOS"]
    run_outlook_shock_playback = ctx["run_outlook_shock_playback"]
    _sector_options = ctx["_sector_options"]
    _top_sector_ticker = ctx["_top_sector_ticker"]
    _build_compare_rows = ctx["_build_compare_rows"]
    _build_counterfactual_row = ctx["_build_counterfactual_row"]
    _build_vulnerability_rows = ctx["_build_vulnerability_rows"]
    _build_why_nodes_rows = ctx["_build_why_nodes_rows"]
    _render_evidence_model_split = ctx["_render_evidence_model_split"]
    build_animated_figure = ctx["build_animated_figure"]

    st.markdown("### 🔮 Systemic Risk Outlook")
    st.caption(
        "Forward-looking monitoring plus scenario-based stress testing. "
        "Use the holdout window to anchor the latest system state, then turn that state into forward stress scenarios."
    )

    min_forecast_date, max_forecast_date = _get_forecast_date_bounds()
    default_test = pd.Timestamp(st.session_state.outlook_test_end or max_forecast_date)
    default_train = pd.Timestamp(
        st.session_state.outlook_train_end
        or forecast_nearest_leq(
            pd.DatetimeIndex(pd.bdate_range(min_forecast_date, max_forecast_date)),
            max_forecast_date - pd.DateOffset(months=3),
        )
    )

    controls = st.columns(7)
    train_end_ui = controls[0].date_input(
        "Train end",
        value=default_train.date(),
        min_value=min_forecast_date.date(),
        max_value=max_forecast_date.date(),
        key="outlook_train_end_ui",
    )
    test_end_ui = controls[1].date_input(
        "Test end",
        value=default_test.date(),
        min_value=min_forecast_date.date(),
        max_value=max_forecast_date.date(),
        key="outlook_test_end_ui",
    )
    walk_step_days_ui = controls[2].selectbox(
        "Walk step",
        options=[5, 10, 20],
        index=[5, 10, 20].index(int(st.session_state.outlook_walk_step_days)),
        key="outlook_walk_step_days_ui",
    )
    walk_horizon_days_ui = controls[3].selectbox(
        "Walk horizon",
        options=[5, 10, 20, 40],
        index=[5, 10, 20, 40].index(int(st.session_state.outlook_walk_horizon_days)),
        key="outlook_walk_horizon_days_ui",
    )
    metric_ui = controls[4].selectbox(
        "Metric",
        options=["density", "avg_abs_weight", "avg_clustering", "risk_pressure", "regime_numeric"],
        index=["density", "avg_abs_weight", "avg_clustering", "risk_pressure", "regime_numeric"].index(
            st.session_state.outlook_metric
        ),
        key="outlook_metric_ui",
    )
    live_outlook_models = ["best", "ridge_regime_aware", "ridge_recursive", "persistence_markov"]
    if st.session_state.outlook_model not in live_outlook_models:
        st.session_state.outlook_model = "best"
    model_ui = controls[5].selectbox(
        "View model",
        options=live_outlook_models,
        index=live_outlook_models.index(st.session_state.outlook_model),
        key="outlook_model_ui",
    )
    run_outlook_btn = controls[6].button("🔮 Run Outlook", use_container_width=True)
    st.session_state.outlook_metric = metric_ui
    st.session_state.outlook_model = model_ui

    if run_outlook_btn:
        train_end_ts = pd.Timestamp(train_end_ui)
        test_end_ts = pd.Timestamp(test_end_ui)
        if train_end_ts >= test_end_ts:
            st.error("`Train end` must be before `Test end`.")
        else:
            with st.spinner("Running outlook backtests..."):
                report_live, joined_live, joined_by_model_live = _run_live_outlook_cached(
                    train_end=str(train_end_ts.date()),
                    test_end=str(test_end_ts.date()),
                    alpha=float(st.session_state.outlook_alpha),
                    walk_step_days=int(walk_step_days_ui),
                    walk_horizon_days=int(walk_horizon_days_ui),
                )
            st.session_state.outlook_report = report_live
            st.session_state.outlook_joined = joined_live
            st.session_state.outlook_joined_by_model = joined_by_model_live
            st.session_state.outlook_metric = metric_ui
            st.session_state.outlook_model = model_ui
            st.session_state.outlook_train_end = str(train_end_ts.date())
            st.session_state.outlook_test_end = str(test_end_ts.date())
            st.session_state.outlook_walk_step_days = int(walk_step_days_ui)
            st.session_state.outlook_walk_horizon_days = int(walk_horizon_days_ui)

    forecast_report = st.session_state.outlook_report
    forecast_joined = st.session_state.outlook_joined
    forecast_joined_by_model = st.session_state.outlook_joined_by_model
    if forecast_report is None:
        forecast_report, forecast_path = load_forecast_report()
        if forecast_joined is not None:
            forecast_joined_by_model = {"best": forecast_joined}
    else:
        forecast_path = _get_forecast_report_path()
    data_info = data_loader.get_data_root_info()

    if forecast_report is None:
        st.info(
            "No forecast result in session yet. You can run the outlook live from this tab, or load an existing report.\n\n"
            f"- Expected report path: `{forecast_path}`\n"
            f"- Current data root: `{data_info['final']}`"
        )
        run_cmd = (
            f"RISKSENTINEL_DATA_ROOT='{data_info['final']}' "
            "./venv/bin/python scripts/run_systemic_risk_forecast.py "
            "--train-end 2025-11-30 --test-end 2026-02-28 "
            "--walk-step-days 20 --walk-horizon-days 20"
        )
        st.code(run_cmd, language="bash")
        return

    fixed = forecast_report.get("fixed_origin") or {}
    walk = forecast_report.get("walk_forward_last_year") or {}
    stress = forecast_report.get("historical_stress_folds") or {}
    summary_rows = _summary_rows_from_forecast(forecast_report)

    top = st.columns(5)
    top[0].metric("Best baseline", str(fixed.get("best_model", fixed.get("model", "n/a"))))
    top[1].metric("Fixed regime acc", f"{100.0 * float((fixed.get('regime') or {}).get('accuracy', 0.0)):.1f}%")
    top[2].metric("Fixed density MAE", f"{float((fixed.get('density') or {}).get('mae', 0.0)):.3f}")
    top[3].metric("Walk folds", str((walk.get("summary") or {}).get("n_folds", 0)))
    top[4].metric("Stress folds", str((stress.get("summary") or {}).get("n_folds", 0)))

    if summary_rows:
        st.markdown("#### Summary Views")
        summary_df = pd.DataFrame(summary_rows)
        for col in ["Regime Acc", "Density MAE", "Risk Pressure MAE", "Top-5 Overlap"]:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].map(
                    lambda x: f"{x:.3f}" if isinstance(x, (int, float, np.floating)) and x is not None else "n/a"
                )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("#### Interpretation")
    st.markdown(
        "- Strongest signal: short-horizon monitoring of aggregate network fragility.\n"
        "- Weakest signal: crisis-state prediction and fine-grained future topology.\n"
        "- Best product framing today: forward-looking systemic surveillance baseline."
    )

    if forecast_joined is not None and not forecast_joined.empty:
        metric_for_plot = metric_ui
        model_for_plot = model_ui
        joined_for_plot = forecast_joined_by_model.get(model_for_plot) if forecast_joined_by_model else None
        if joined_for_plot is None:
            joined_for_plot = forecast_joined
        holdout_dates = [str(ts.date()) for ts in joined_for_plot.index]
        default_focus_date = st.session_state.outlook_focus_date or holdout_dates[-1]
        if default_focus_date not in holdout_dates:
            default_focus_date = holdout_dates[-1]
        focus_date_ui = st.selectbox(
            "Focus date",
            options=holdout_dates,
            index=holdout_dates.index(default_focus_date),
            key="outlook_focus_date_ui",
        )
        st.session_state.outlook_focus_date = focus_date_ui
        snapshot = _compute_outlook_snapshot(joined_for_plot, focus_date_ui)
        change_rows = _build_change_rows(joined_for_plot, focus_date_ui)
        monitor_subtab, stress_subtab = st.tabs(["📡 Monitor", "💥 Stress Test"])

        with monitor_subtab:
            st.markdown("#### Forward Stress Monitor")
            system_cols = st.columns(3)
            system_cols[0].markdown(
                (
                    '<div class="insight-card">'
                    '<div class="insight-kicker">Current Regime</div>'
                    f'<div class="insight-value">{html.escape(str(snapshot.get("regime_label", "Unknown")))}</div>'
                    f'<div class="insight-copy">{_regime_pill_html(str(snapshot.get("regime_label", "Unknown")))}'
                    f'Observed on {html.escape(str(snapshot.get("focus_date", "n/a")))}.</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            system_cols[1].markdown(
                (
                    '<div class="insight-card">'
                    '<div class="insight-kicker">Fragility Trend</div>'
                    f'<div class="insight-value">{html.escape(str(snapshot.get("fragility_trend", "Stable")))}</div>'
                    f'<div class="insight-copy">Risk pressure moved '
                    f'{100.0 * float(snapshot.get("risk_delta_pct", 0.0)):+.1f}% versus {html.escape(str(snapshot.get("previous_date", "n/a")))}.'
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            system_cols[2].markdown(
                (
                    '<div class="insight-card">'
                    '<div class="insight-kicker">Forward Stress Readiness</div>'
                    f'<div class="insight-value">{html.escape(str(snapshot.get("stress_readiness", "Ready")))}</div>'
                    f'<div class="insight-copy">Density {float(snapshot.get("density", 0.0)):.3f} | '
                    f'Risk pressure {float(snapshot.get("risk_pressure", 0.0)):.3f}.</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='insight-copy'>Regime bands: "
                + " ".join(_regime_pill_html(label) for label in ["Calm", "Normal", "Elevated", "High", "Crisis"])
                + " shaded behind the monitoring path.</div>",
                unsafe_allow_html=True,
            )
            status_html = (
                "<div class='insight-copy'>"
                f"{_regime_pill_html(str(snapshot.get('regime_label', 'Unknown')))}"
                f"{_regime_pill_html(str(snapshot.get('fragility_trend', 'Stable')))}"
                f"{_regime_pill_html(str(snapshot.get('stress_readiness', 'Ready')))}"
                "</div>"
            )
            st.markdown(status_html, unsafe_allow_html=True)
            meta_cols = st.columns(2)
            meta_cols[0].markdown(
                (
                    '<div class="insight-card">'
                    '<div class="insight-kicker">Regime Transition</div>'
                    f'<div class="insight-copy">{html.escape(_build_regime_transition_copy(snapshot))}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            meta_cols[1].markdown(
                (
                    '<div class="insight-card">'
                    '<div class="insight-kicker">Forecast Confidence</div>'
                    f'<div class="insight-copy">{html.escape(_forecast_confidence_copy(forecast_report, metric_for_plot))}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                f"Viewing {_format_outlook_metric_label(metric_for_plot)} with `{model_for_plot}` across the selected holdout window. "
                f"`Focus date` anchors both the monitoring view and shock playback."
            )
            chart_cols = st.columns([1.35, 1.0])
            compact_fig = build_outlook_compact_figure(joined_for_plot, metric_for_plot, focus_date=focus_date_ui)
            if compact_fig:
                chart_cols[0].plotly_chart(compact_fig, use_container_width=True, config={"displayModeBar": False})
            spread_fig = build_outlook_spread_figure(joined_for_plot, metric_for_plot, focus_date=focus_date_ui)
            if spread_fig:
                chart_cols[1].plotly_chart(spread_fig, use_container_width=True, config={"displayModeBar": False})
            checkpoint_rows = _build_outlook_checkpoint_rows(joined_for_plot, metric_for_plot)
            if checkpoint_rows:
                st.markdown("#### Checkpoint Outlook")
                st.dataframe(pd.DataFrame(checkpoint_rows), use_container_width=True, hide_index=True)
            with st.expander("Playback View", expanded=False):
                animate_ui = st.toggle("Animate", value=False, key="outlook_animate_ui")
                fig_outlook = (
                    build_outlook_animation_figure(joined_for_plot, metric_for_plot, focus_date=focus_date_ui)
                    if animate_ui
                    else build_outlook_timeseries_figure(joined_for_plot, metric_for_plot, focus_date=focus_date_ui)
                )
                if fig_outlook:
                    st.plotly_chart(fig_outlook, use_container_width=True, config={"displayModeBar": False})
            narrative_lines = _build_narrative_lines(snapshot, _top_systemic_rows(focus_date_ui, limit=5), None)
            if narrative_lines:
                st.markdown("#### Supervisory Readout")
                for line in narrative_lines:
                    st.markdown(f"- {line}")
            action_cols = st.columns(2)
            action_cols[0].markdown("#### Recommended Actions")
            action_cols[0].dataframe(
                pd.DataFrame(_build_action_rows(snapshot, st.session_state.outlook_shock_bundle)),
                use_container_width=True,
                hide_index=True,
            )
            action_cols[1].markdown("#### Why This Matters")
            action_cols[1].dataframe(
                pd.DataFrame(_build_why_this_matters_rows(snapshot)),
                use_container_width=True,
                hide_index=True,
            )
            if change_rows:
                st.markdown("#### What Changed Since Last Month")
                st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)
            watchlist_rows = _build_watchlist_rows(focus_date_ui, limit=8)
            if watchlist_rows:
                st.markdown("#### Watchlist Panel")
                st.dataframe(pd.DataFrame(watchlist_rows), use_container_width=True, hide_index=True)
            out_cols = st.columns(2)
            out_cols[0].download_button(
                "📄 Outlook JSON",
                json.dumps(forecast_report, indent=2),
                file_name="risksentinel_outlook_report.json",
                mime="application/json",
                use_container_width=True,
            )
            out_cols[1].download_button(
                "📊 Outlook CSV",
                joined_for_plot.to_csv(),
                file_name="risksentinel_outlook_timeseries.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with stress_subtab:
            st.markdown("#### Scenario Cards")
            scenario_cols = st.columns(3)
            for idx, scenario in enumerate(OUTLOOK_SCENARIOS):
                scenario_ticker = scenario["ticker"] if scenario["ticker"] in st.session_state.tickers else st.session_state.tickers[0]
                scenario_cols[idx].markdown(
                    (
                        '<div class="scenario-card">'
                        f'<div class="scenario-title">{html.escape(scenario["title"])}</div>'
                        f'<div class="scenario-copy">{html.escape(scenario["copy"])}</div>'
                        f'<div class="scenario-meta">{html.escape(scenario_ticker)} | '
                        f'{int(scenario["shock_pct"])}% shock | {html.escape(scenario["model"])} | '
                        f'threshold {float(scenario["threshold"]):.2f}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if scenario_cols[idx].button(
                    f"Run {scenario['title']}",
                    key=f"outlook_scenario_{scenario['key']}",
                    use_container_width=True,
                ):
                    st.session_state.outlook_sync_shock_date = True
                    st.session_state.outlook_shock_date = focus_date_ui
                    st.session_state.outlook_shock_ticker = scenario_ticker
                    st.session_state.outlook_shock_pct = int(scenario["shock_pct"])
                    st.session_state.outlook_shock_model = str(scenario["model"])
                    st.session_state.outlook_shock_threshold = float(scenario["threshold"])
                    st.session_state.outlook_policy_intervention = "none"
                    try:
                        with st.spinner(f"Running {scenario['title'].lower()} scenario..."):
                            st.session_state.outlook_shock_bundle = run_outlook_shock_playback(
                                date_str=focus_date_ui,
                                threshold=float(scenario["threshold"]),
                                ticker=scenario_ticker,
                                shock_pct=int(scenario["shock_pct"]),
                                model=str(scenario["model"]),
                                intervention="none",
                            )
                    except Exception as exc:
                        st.session_state.outlook_shock_bundle = None
                        st.error(str(exc))

            st.markdown("#### Forward Stress Test")
            config_cols = st.columns(4)
            seed_mode_ui = config_cols[0].selectbox(
                "Seed mode",
                options=["ticker", "sector"],
                index=["ticker", "sector"].index(st.session_state.outlook_shock_seed_mode),
                key="outlook_shock_seed_mode_ui",
            )
            sector_options = _sector_options() or ["Financials"]
            sector_ui = config_cols[1].selectbox(
                "Sector seed",
                options=sector_options,
                index=sector_options.index(st.session_state.outlook_shock_sector)
                if st.session_state.outlook_shock_sector in sector_options
                else 0,
                disabled=seed_mode_ui != "sector",
                key="outlook_shock_sector_ui",
            )
            intervention_options = ["none", "remove_top_connector", "cap_exposure", "sector_firebreak"]
            intervention_ui = config_cols[2].selectbox(
                "Policy intervention",
                options=intervention_options,
                index=intervention_options.index(st.session_state.outlook_policy_intervention)
                if st.session_state.outlook_policy_intervention in intervention_options
                else 0,
                key="outlook_policy_intervention_ui",
            )
            compare_enabled_ui = config_cols[3].toggle(
                "Scenario compare",
                value=bool(st.session_state.outlook_compare_enabled),
                key="outlook_compare_enabled_ui",
            )
            st.session_state.outlook_shock_seed_mode = seed_mode_ui
            st.session_state.outlook_shock_sector = sector_ui
            st.session_state.outlook_policy_intervention = intervention_ui
            st.session_state.outlook_compare_enabled = bool(compare_enabled_ui)
            default_shock_date = st.session_state.outlook_shock_date or holdout_dates[-1]
            if default_shock_date not in holdout_dates:
                default_shock_date = holdout_dates[-1]
            shock_cols = st.columns(7)
            sync_shock_date_ui = shock_cols[0].toggle(
                "Sync to focus",
                value=bool(st.session_state.outlook_sync_shock_date),
                key="outlook_sync_shock_date_ui",
            )
            resolved_shock_date = focus_date_ui if sync_shock_date_ui else default_shock_date
            shock_date_ui = shock_cols[1].selectbox(
                "Shock date",
                options=holdout_dates,
                index=holdout_dates.index(resolved_shock_date),
                disabled=sync_shock_date_ui,
                key="outlook_shock_date_ui",
            )
            shock_ticker_ui = shock_cols[2].selectbox(
                "Shock ticker",
                options=st.session_state.tickers,
                index=st.session_state.tickers.index(st.session_state.outlook_shock_ticker)
                if st.session_state.outlook_shock_ticker in st.session_state.tickers
                else 0,
                key="outlook_shock_ticker_ui",
            )
            shock_pct_ui = shock_cols[3].slider(
                "Shock %",
                min_value=10,
                max_value=100,
                value=int(st.session_state.outlook_shock_pct),
                step=10,
                key="outlook_shock_pct_ui",
            )
            shock_model_ui = shock_cols[4].selectbox(
                "Shock model",
                options=["debtrank", "linear_threshold", "cascade_removal"],
                index=["debtrank", "linear_threshold", "cascade_removal"].index(st.session_state.outlook_shock_model),
                key="outlook_shock_model_ui",
            )
            shock_threshold_ui = shock_cols[5].slider(
                "Corr. threshold",
                min_value=0.2,
                max_value=0.8,
                value=float(st.session_state.outlook_shock_threshold),
                step=0.05,
                key="outlook_shock_threshold_ui",
            )
            run_shock_playback_btn = shock_cols[6].button("💥 Run Stress Playback", use_container_width=True)

            st.session_state.outlook_sync_shock_date = bool(sync_shock_date_ui)
            st.session_state.outlook_shock_date = focus_date_ui if sync_shock_date_ui else shock_date_ui
            st.session_state.outlook_shock_ticker = shock_ticker_ui
            st.session_state.outlook_shock_pct = int(shock_pct_ui)
            st.session_state.outlook_shock_model = shock_model_ui
            st.session_state.outlook_shock_threshold = float(shock_threshold_ui)
            primary_ticker = _top_sector_ticker(focus_date_ui, sector_ui) if seed_mode_ui == "sector" else shock_ticker_ui

            if run_shock_playback_btn:
                try:
                    with st.spinner("Running shock playback..."):
                        st.session_state.outlook_shock_bundle = run_outlook_shock_playback(
                            date_str=st.session_state.outlook_shock_date,
                            threshold=float(shock_threshold_ui),
                            ticker=primary_ticker,
                            shock_pct=int(shock_pct_ui),
                            model=shock_model_ui,
                            intervention=intervention_ui,
                        )
                        if compare_enabled_ui:
                            compare_ticker = st.session_state.outlook_compare_ticker
                            compare_model = st.session_state.outlook_compare_model
                            compare_shock = int(st.session_state.outlook_compare_shock_pct)
                            st.session_state.outlook_compare_bundle = run_outlook_shock_playback(
                                date_str=st.session_state.outlook_shock_date,
                                threshold=float(shock_threshold_ui),
                                ticker=compare_ticker,
                                shock_pct=compare_shock,
                                model=compare_model,
                                intervention="none",
                            )
                        else:
                            st.session_state.outlook_compare_bundle = None
                except Exception as exc:
                    st.session_state.outlook_shock_bundle = None
                    st.session_state.outlook_compare_bundle = None
                    st.error(str(exc))

            shock_bundle = st.session_state.outlook_shock_bundle
            compare_bundle = st.session_state.outlook_compare_bundle
            if compare_enabled_ui:
                compare_cols = st.columns(3)
                compare_cols[0].selectbox(
                    "Compare ticker",
                    options=st.session_state.tickers,
                    index=st.session_state.tickers.index(st.session_state.outlook_compare_ticker)
                    if st.session_state.outlook_compare_ticker in st.session_state.tickers
                    else 0,
                    key="outlook_compare_ticker_ui",
                )
                compare_cols[1].selectbox(
                    "Compare model",
                    options=["debtrank", "linear_threshold", "cascade_removal"],
                    index=["debtrank", "linear_threshold", "cascade_removal"].index(st.session_state.outlook_compare_model),
                    key="outlook_compare_model_ui",
                )
                compare_cols[2].slider(
                    "Compare shock %",
                    min_value=10,
                    max_value=100,
                    value=int(st.session_state.outlook_compare_shock_pct),
                    step=10,
                    key="outlook_compare_shock_pct_ui",
                )
                st.session_state.outlook_compare_ticker = st.session_state.outlook_compare_ticker_ui
                st.session_state.outlook_compare_model = st.session_state.outlook_compare_model_ui
                st.session_state.outlook_compare_shock_pct = int(st.session_state.outlook_compare_shock_pct_ui)

            if shock_bundle:
                shock_meta_cols = st.columns(4)
                shock_summary = shock_bundle["result"].summary()
                shock_meta_cols[0].metric("Playback date", str(shock_bundle["date"]))
                shock_meta_cols[1].metric("Affected", int(shock_summary["n_affected"]))
                shock_meta_cols[2].metric("Cascade", f"{shock_summary['cascade_depth']} waves")
                shock_meta_cols[3].metric("Avg stress", f"{100.0 * float(shock_summary['avg_stress']):.1f}%")

                blast_playback_ui = st.toggle(
                    "Blast radius only",
                    value=False,
                    key="outlook_blast_radius_playback_ui",
                )
                shock_fig = build_animated_figure(
                    shock_bundle["G"],
                    shock_bundle["pos"],
                    shock_bundle["result"],
                    blast_radius_only=blast_playback_ui,
                )
                st.plotly_chart(
                    shock_fig,
                    use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": True},
                    key="outlook_shock_playback_chart",
                )
                st.caption(
                    f"Shock playback uses the observed network nearest to `{shock_bundle['date']}` with "
                    f"`{shock_bundle['model']}` on `{shock_bundle['ticker']}` at {shock_bundle['shock_pct']}% shock. "
                    f"Policy: `{(shock_bundle.get('intervention_meta') or {}).get('label', 'No intervention')}`."
                )
            else:
                st.info("Pick a scenario card or run a custom stress test to populate playback, compare, and explainability panels.")

            if compare_enabled_ui:
                st.markdown("#### Scenario Compare")
                st.dataframe(
                    pd.DataFrame(_build_compare_rows(shock_bundle, compare_bundle)),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Avg Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")},
                )
                counterfactual = _build_counterfactual_row(shock_bundle, compare_bundle)
                if counterfactual:
                    cf_cols = st.columns(4)
                    cf_cols[0].metric("Affected delta", counterfactual["Affected delta"])
                    cf_cols[1].metric("Wave delta", counterfactual["Wave delta"])
                    cf_cols[2].metric("Avg stress delta", f"{counterfactual['Avg stress delta %']:+.1f}%")
                    cf_cols[3].metric("Total stress delta", f"{counterfactual['Total stress delta']:+.3f}")
            vulnerability_rows = _build_vulnerability_rows(
                st.session_state.outlook_shock_date or focus_date_ui,
                shock_bundle,
                limit=8,
            )
            st.markdown("#### Top Vulnerable Nodes")
            vuln_config = {}
            if shock_bundle:
                vuln_config["Shock Stress %"] = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")
            st.dataframe(
                pd.DataFrame(vulnerability_rows),
                use_container_width=True,
                hide_index=True,
                column_config=vuln_config,
            )
            narrative_lines = _build_narrative_lines(snapshot, vulnerability_rows, shock_bundle)
            if narrative_lines:
                st.markdown("#### Intervention Readout")
                for line in narrative_lines:
                    st.markdown(f"- {line}")
            why_nodes_rows = _build_why_nodes_rows(st.session_state.outlook_shock_date or focus_date_ui, shock_bundle, limit=6)
            if why_nodes_rows:
                st.markdown("#### Why These Nodes")
                st.dataframe(
                    pd.DataFrame(why_nodes_rows),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")},
                )
            st.markdown("#### Evidence vs Model")
            _render_evidence_model_split()

    if fixed:
        st.markdown("#### Fixed-Origin Holdout")
        fixed_cols = st.columns(4)
        fixed_cols[0].metric("Regime acc", f"{100.0 * float((fixed.get('regime') or {}).get('accuracy', 0.0)):.1f}%")
        fixed_cols[1].metric("Density MAE", f"{float((fixed.get('density') or {}).get('mae', 0.0)):.4f}")
        fixed_cols[2].metric("Avg |weight| MAE", f"{float((fixed.get('avg_abs_weight') or {}).get('mae', 0.0)):.4f}")
        fixed_cols[3].metric(
            "Top-5 overlap",
            f"{100.0 * float((fixed.get('top_systemic_nodes_persistence') or {}).get('mean_overlap', 0.0)):.1f}%"
        )
        examples = fixed.get("examples") or []
        if examples:
            st.dataframe(pd.DataFrame(examples), use_container_width=True, hide_index=True)

    walk_summary = walk.get("summary") or {}
    if walk_summary:
        st.markdown("#### Walk-Forward Last Year")
        walk_cols = st.columns(4)
        walk_cols[0].metric("Mean regime acc", f"{100.0 * float(walk_summary.get('regime_accuracy_mean', 0.0)):.1f}%")
        walk_cols[1].metric("Mean density MAE", f"{float(walk_summary.get('density_mae_mean', 0.0)):.4f}")
        walk_cols[2].metric("Mean risk pressure MAE", f"{float(walk_summary.get('risk_pressure_mae_mean', 0.0)):.4f}")
        walk_cols[3].metric("Mean top-5 overlap", f"{100.0 * float(walk_summary.get('top_k_overlap_mean', 0.0)):.1f}%")
        st.caption(
            f"Eval window: {walk.get('eval_start', 'n/a')} -> {walk.get('eval_end', 'n/a')} | "
            f"step {walk.get('step_days', 'n/a')} days | horizon {walk.get('horizon_days', 'n/a')} days"
        )
        with st.expander("Walk-forward fold details", expanded=False):
            fold_rows = []
            for fold in walk.get("folds") or []:
                fold_rows.append(
                    {
                        "train_end": fold.get("train_end"),
                        "test_end": fold.get("test_end"),
                        "model": fold.get("best_model", fold.get("model")),
                        "regime_acc": (fold.get("regime") or {}).get("accuracy"),
                        "density_mae": (fold.get("density") or {}).get("mae"),
                        "top_5_overlap": (fold.get("top_systemic_nodes_persistence") or {}).get("mean_overlap"),
                    }
                )
            if fold_rows:
                st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)

    stress_summary = stress.get("summary") or {}
    if stress_summary:
        st.markdown("#### Historical Stress Validation")
        stress_cols = st.columns(4)
        stress_cols[0].metric("Mean regime acc", f"{100.0 * float(stress_summary.get('regime_accuracy_mean', 0.0)):.1f}%")
        stress_cols[1].metric("Mean density MAE", f"{float(stress_summary.get('density_mae_mean', 0.0)):.4f}")
        stress_cols[2].metric("Mean risk pressure MAE", f"{float(stress_summary.get('risk_pressure_mae_mean', 0.0)):.4f}")
        stress_cols[3].metric("Mean top-5 overlap", f"{100.0 * float(stress_summary.get('top_k_overlap_mean', 0.0)):.1f}%")
        with st.expander("Stress fold details", expanded=False):
            stress_rows = []
            for fold in stress.get("folds") or []:
                stress_rows.append(
                    {
                        "fold": fold.get("fold_label", "n/a"),
                        "train_end": fold.get("train_end"),
                        "test_end": fold.get("test_end"),
                        "model": fold.get("best_model", fold.get("model")),
                        "regime_acc": (fold.get("regime") or {}).get("accuracy"),
                        "density_mae": (fold.get("density") or {}).get("mae"),
                        "top_5_overlap": (fold.get("top_systemic_nodes_persistence") or {}).get("mean_overlap"),
                    }
                )
            if stress_rows:
                st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)

    with st.expander("Raw forecast report", expanded=False):
        st.json(forecast_report)
