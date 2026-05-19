"""App-shell flow helpers for deterministic sidebar actions."""

from __future__ import annotations


def snapshot_local_state(session_state) -> dict[str, object]:
    return {
        "graph_data": session_state.graph_data,
        "pos": session_state.pos,
        "shock_result": session_state.shock_result,
        "current_wave": session_state.current_wave,
        "agent_messages": list(session_state.agent_messages),
        "comparison": session_state.comparison,
    }


def restore_local_state(session_state, snapshot: dict[str, object]) -> None:
    session_state.graph_data = snapshot["graph_data"]
    session_state.pos = snapshot["pos"]
    session_state.shock_result = snapshot["shock_result"]
    session_state.current_wave = snapshot["current_wave"]
    session_state.agent_messages = snapshot["agent_messages"]
    session_state.comparison = snapshot["comparison"]


def handle_preset_trigger(session_state, params: dict[str, object], build_fn, shock_fn) -> None:
    session_state.sel_date = params["date"]
    session_state.sel_ticker = params["ticker"]
    session_state.sel_shock = params["shock"]
    session_state.sel_threshold = params["threshold"]
    session_state.agent_messages = []
    graph = build_fn(params["date"], params["threshold"])
    shock_fn(graph, params["ticker"], params["shock"], "debtrank")


def ensure_agentic_context(session_state, data_loader_obj, requested: bool) -> tuple[dict[str, str], list[str], str | None]:
    sector_dict_ctx = dict(session_state.get("sector_dict") or {})
    tickers_ctx = list(session_state.get("tickers") or [])
    if requested and (not sector_dict_ctx or not tickers_ctx):
        try:
            sector_dict_ctx = dict(data_loader_obj.get_sector_dict() or {})
            tickers_ctx = list(data_loader_obj.get_ticker_list() or [])
            session_state.sector_dict = sector_dict_ctx
            session_state.tickers = tickers_ctx
        except Exception as exc:
            return {}, [], f"Agentic ops unavailable: reference data not loaded ({type(exc).__name__}: {exc})."
    return sector_dict_ctx, tickers_ctx, None


def run_build_action(session_state, selected_date: str, threshold: float, build_fn) -> None:
    session_state.agent_messages = []
    build_fn(selected_date, threshold)


def run_shock_action(session_state, shocked_ticker: str, shock_pct: int, shock_model: str, shock_fn) -> None:
    if not session_state.graph_data:
        return
    session_state.comparison = None
    shock_fn(session_state.graph_data["G"], shocked_ticker, shock_pct, shock_model)


def run_compare_action(session_state, shocked_ticker: str, shock_pct: int, contagion_module) -> None:
    if not session_state.graph_data:
        return
    graph = session_state.graph_data["G"]
    if shocked_ticker not in graph:
        return
    session_state.comparison = contagion_module.compare_models(graph, shocked_ticker, shock_pct / 100.0)
    session_state.shock_result = session_state.comparison["debtrank"]
    session_state.current_wave = session_state.shock_result.cascade_depth
    session_state.agent_messages = [
        (
            "Quant",
            "📊",
            "agent-quant",
            f"<b>Model Comparison</b> — {shocked_ticker} at {shock_pct}% shock. See comparison table below the network graph.",
        )
    ]
