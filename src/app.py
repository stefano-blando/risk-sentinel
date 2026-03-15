"""
RiskSentinel — Streamlit App
Agentic Systemic Risk Simulator for Financial Contagion
"""

import concurrent.futures
import sys
import html
import time
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx

from src.agents.evidence_rag import (
    build_crisis_evidence_docs,
    build_history_evidence_docs,
    format_evidence_block,
    retrieve_evidence,
    serialize_retrieved,
)
from src.agents.evidence_validation import validate_payload_evidence
from src.core import data_loader, network, contagion
from src.core.forecasting import (
    nearest_leq as forecast_nearest_leq,
)
from src import agentic_ops, ui_panels
from src.ui.charts.outlook import (
    build_outlook_animation_figure as build_outlook_animation_figure_chart,
    build_outlook_checkpoint_rows,
    build_outlook_compact_figure as build_outlook_compact_figure_chart,
    build_outlook_spread_figure as build_outlook_spread_figure_chart,
    build_outlook_timeseries_figure as build_outlook_timeseries_figure_chart,
    format_outlook_metric_label,
)
from src.ui.charts.network import (
    build_animated_figure as build_animated_figure_chart,
    build_graph_figure as build_graph_figure_chart,
    compute_layout,
)
from src.ui.charts.surveillance import (
    build_sector_impact_bar_figure as build_sector_impact_bar_figure_chart,
    build_stress_tier_donut_figure as build_stress_tier_donut_figure_chart,
    build_systemic_risk_gauge_figure as build_systemic_risk_gauge_figure_chart,
    build_timeline_figure as build_timeline_figure_chart,
    build_wave_trend_figure as build_wave_trend_figure_chart,
)
from src.ui.services.outlook import (
    OUTLOOK_SCENARIOS,
    build_action_rows,
    build_change_rows,
    build_compare_rows,
    build_counterfactual_row,
    build_narrative_lines,
    build_regime_transition_copy,
    build_vulnerability_rows,
    build_watchlist_rows,
    build_why_nodes_rows,
    build_why_this_matters_rows,
    compute_outlook_snapshot,
    forecast_confidence_copy,
    get_forecast_date_bounds,
    top_systemic_rows,
    run_live_outlook_cached,
    summary_rows_from_forecast,
)
from src.ui.services.audit_trail import (
    build_judge_kpis,
    build_judge_run_rows,
    build_submission_bundle_bytes,
    generate_action_pack_ceo_brief,
    generate_action_pack_machine_json,
    generate_action_pack_runbook,
    generate_report_markdown,
    generate_report_text,
    generate_trace_bundle_json,
    summarize_quality,
)
from src.ui.services.agentic_actions import run_sidebar_agentic_actions
from src.ui.services.agentic_domain import (
    build_agent_cache_key as build_agent_cache_key_service,
    build_context_facts_html as build_context_facts_html_service,
    build_memory_hint as build_memory_hint_service,
    build_session_decision_hint as build_session_decision_hint_service,
    build_structured_prompt as build_structured_prompt_service,
    evaluate_run_trace as evaluate_run_trace_service,
    find_cached_agent_response as find_cached_agent_response_service,
    format_llm_text_for_card as format_llm_text_for_card_service,
    get_agent_config_status as get_agent_config_status_service,
    remember_session_decision as remember_session_decision_service,
)
from src.ui.services.app_flows import (
    ensure_agentic_context,
    handle_preset_trigger,
    restore_local_state,
    run_build_action,
    run_compare_action,
    run_shock_action,
    snapshot_local_state,
)
from src.ui.services.evaluation import (
    run_local_benchmark as run_local_benchmark_service,
    run_scenario_pack_eval as run_scenario_pack_eval_service,
)
from src.ui.services.query_ops import (
    build_cache_fingerprint,
    extract_json_payload,
    infer_model_from_query,
    is_compare_query,
    is_complex_query,
    is_query_in_scope,
    jaccard_similarity as jaccard_similarity_service,
    normalize_chat_query,
    parse_chat_query as parse_chat_query_service,
    parse_structured_agent_output as parse_structured_agent_output_service,
    render_structured_payload_html,
    tokenize_query as tokenize_query_service,
    extract_tickers_from_query as extract_tickers_from_query_service,
)
from src.ui.services.runtime import (
    agentic_cache_key as agentic_cache_key_service,
    check_gpt_rate_limit as check_gpt_rate_limit_service,
    estimate_eta_seconds as estimate_eta_seconds_service,
    get_gpt_access_policy as get_gpt_access_policy_service,
    get_runtime_int as get_runtime_int_service,
    get_runtime_value as get_runtime_value_service,
    is_gpt_circuit_open as is_gpt_circuit_open_service,
    prune_events as prune_events_service,
    register_gpt_call as register_gpt_call_service,
    register_gpt_failure as register_gpt_failure_service,
    register_gpt_success as register_gpt_success_service,
    run_agentic_operation as run_agentic_operation_service,
    unlock_judge_access as unlock_judge_access_service,
)
from src.ui.services.tracing import (
    advance_workflow as advance_workflow_service,
    create_run_trace as create_run_trace_service,
    finalize_run_trace as finalize_run_trace_service,
    persist_run_trace as persist_run_trace_service,
    trace_event as trace_event_service,
)
from src.ui.services.llm_ops import (
    get_deployment_routing as get_deployment_routing_service,
    is_rate_limit_error as is_rate_limit_error_service,
    is_retryable_gpt_error as is_retryable_gpt_error_service,
    is_timeout_error as is_timeout_error_service,
    run_agent_query as run_agent_query_service,
    run_agent_query_with_backoff as run_agent_query_with_backoff_service,
    run_critic_validation as run_critic_validation_service,
    run_gpt_diagnostic as run_gpt_diagnostic_service,
)
from src.ui.services.surveillance import build_severity_df
from src.ui.services.stress_lab import build_compare_rows_df
from src.ui.services.simulation import (
    build_compare_facts_html as build_compare_facts_html_service,
    build_simulation_facts_html as build_simulation_facts_html_service,
    compute_compare_rows,
    execute_build_network,
    execute_shock_scenario,
)
from src.ui.sidebar import render_sidebar
from src.ui.state import build_app_defaults
from src.ui.tabs.audit_trail import render_tab as render_audit_trail_tab
from src.ui.tabs.stress_lab import render_tab as render_stress_lab_tab
from src.ui.tabs.surveillance import render_tab as render_surveillance_tab
from src.ui.tabs.outlook import render_tab as render_outlook_tab
from src.utils.azure_config import (
    get_settings,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RiskSentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME TOKENS
# ---------------------------------------------------------------------------
PALETTE = {
    "bg_main": "#070b14",
    "surface_0": "#0f172a",
    "surface_1": "#131f37",
    "surface_2": "#1a2744",
    "border": "#223555",
    "text_primary": "#e5edf9",
    "text_muted": "#9fb0cc",
    "accent_cool": "#38bdf8",
    "accent_blue": "#60a5fa",
    "accent_warm": "#fb923c",
    "accent_hot": "#ef4444",
    "success": "#22c55e",
    "warning": "#fbbf24",
    "elevated": "#f97316",
    "danger": "#f87171",
}

AGENT_BORDER_COLORS = {
    "architect": PALETTE["accent_cool"],
    "quant": PALETTE["accent_warm"],
    "advisor": "#34d399",
    "sentinel": "#c084fc",
}

RISK_COLORS = {
    "critical": "#ef4444",
    "high": "#f97316",
    "moderate": "#fbbf24",
    "low": PALETTE["accent_cool"],
    "none": "#64748b",
}

MODEL_COLORS = {
    "Linear Threshold": PALETTE["accent_cool"],
    "Debtrank": PALETTE["accent_warm"],
    "Cascade Removal": AGENT_BORDER_COLORS["sentinel"],
}

PLOT_EDGE_BG = "rgba(73, 95, 132, 0.28)"
PLOT_EDGE_STRESS = "rgba(249, 115, 22, 0.46)"
PLOT_EVENT_FILL = "rgba(239, 68, 68, 0.12)"
PLOTLY_STRESS_COLORSCALE = [
    [0.0, PALETTE["accent_cool"]],
    [0.4, PALETTE["warning"]],
    [0.7, PALETTE["accent_warm"]],
    [1.0, PALETTE["accent_hot"]],
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    :root {{
        --bg-main: {PALETTE["bg_main"]};
        --surface-0: {PALETTE["surface_0"]};
        --surface-1: {PALETTE["surface_1"]};
        --surface-2: {PALETTE["surface_2"]};
        --border-subtle: {PALETTE["border"]};
        --text-primary: {PALETTE["text_primary"]};
        --text-muted: {PALETTE["text_muted"]};
        --accent-cool: {PALETTE["accent_cool"]};
        --accent-blue: {PALETTE["accent_blue"]};
        --accent-warm: {PALETTE["accent_warm"]};
        --accent-hot: {PALETTE["accent_hot"]};
    }}
    .stApp {{
        background:
            radial-gradient(circle at 10% -10%, rgba(56, 189, 248, 0.18) 0%, transparent 35%),
            radial-gradient(circle at 90% 12%, rgba(251, 146, 60, 0.10) 0%, transparent 32%),
            var(--bg-main);
        color: var(--text-primary);
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface-0) 0%, #101b30 100%);
        border-right: 1px solid var(--border-subtle);
    }}
    [data-testid="stSidebar"] * {{
        color: var(--text-primary);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.75rem;
        margin-bottom: 0.6rem;
        flex-wrap: wrap;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: var(--surface-0);
        border: 1px solid var(--border-subtle);
        border-bottom: none;
        border-radius: 10px 10px 0 0;
        color: var(--text-muted);
        padding: 0.5rem 0.95rem;
        min-height: 42px;
        min-width: 118px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #16305d 0%, #1c3f79 100%);
        color: var(--text-primary);
        border-color: #355c97;
    }}
    div[data-testid="stMetric"] {{
        background: var(--surface-1);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 10px 12px;
    }}
    div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {{
        color: var(--text-muted);
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--text-primary);
    }}
    .stButton > button, .stDownloadButton > button {{
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        background: var(--surface-1);
        color: var(--text-primary);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: var(--accent-blue);
        background: #192949;
        color: #ffffff;
    }}
    .stExpander {{
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
    }}
    .agent-msg {{
        background: var(--surface-1);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
    }}
    .agent-architect {{ border-color: {AGENT_BORDER_COLORS["architect"]}; }}
    .agent-quant {{ border-color: {AGENT_BORDER_COLORS["quant"]}; }}
    .agent-advisor {{ border-color: {AGENT_BORDER_COLORS["advisor"]}; }}
    .agent-sentinel {{ border-color: {AGENT_BORDER_COLORS["sentinel"]}; }}
    .risk-low {{ color: {PALETTE["success"]}; }}
    .risk-moderate {{ color: {PALETTE["warning"]}; }}
    .risk-elevated {{ color: {PALETTE["elevated"]}; }}
    .risk-high {{ color: {PALETTE["danger"]}; }}
    .risk-critical {{ color: {RISK_COLORS["critical"]}; font-weight: bold; }}
    .insight-card {{
        background: linear-gradient(180deg, rgba(20, 32, 56, 0.98) 0%, rgba(12, 20, 36, 0.98) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 14px 16px;
        min-height: 128px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
    }}
    .insight-kicker {{
        color: var(--text-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }}
    .insight-value {{
        color: var(--text-primary);
        font-size: 28px;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 8px;
    }}
    .insight-copy {{
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.45;
    }}
    .status-pill {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .pill-calm {{ background: rgba(34, 197, 94, 0.14); color: #86efac; }}
    .pill-normal {{ background: rgba(56, 189, 248, 0.14); color: #7dd3fc; }}
    .pill-elevated {{ background: rgba(251, 191, 36, 0.14); color: #fcd34d; }}
    .pill-high {{ background: rgba(251, 146, 60, 0.14); color: #fdba74; }}
    .pill-crisis {{ background: rgba(239, 68, 68, 0.16); color: #fca5a5; }}
    .scenario-card {{
        background: linear-gradient(180deg, rgba(17, 28, 48, 0.98) 0%, rgba(10, 17, 31, 0.98) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 14px 16px 10px 16px;
        min-height: 150px;
    }}
    .scenario-title {{
        color: var(--text-primary);
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 6px;
    }}
    .scenario-copy {{
        color: var(--text-muted);
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 10px;
    }}
    .scenario-meta {{
        color: var(--text-muted);
        font-size: 12px;
    }}
    .main-header {{
        background: linear-gradient(110deg, #123568 0%, #1a4f92 45%, #0f6ea2 100%);
        border: 1px solid #2a5d99;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }}
    .sector-dot {{
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }}
    .crisis-btn button {{
        font-size: 12px !important;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CRISIS PRESETS
# ---------------------------------------------------------------------------
CRISIS_PRESETS = {
    "COVID-19 Crash": {"date": "2020-03-16", "ticker": "BAC", "shock": 60, "threshold": 0.75},
    "SVB Crisis": {"date": "2023-03-13", "ticker": "SCHW", "shock": 50, "threshold": 0.6},
    "Japan Carry Trade": {"date": "2024-08-05", "ticker": "GS", "shock": 40, "threshold": 0.6},
    "Volmageddon 2018": {"date": "2018-02-08", "ticker": "JPM", "shock": 40, "threshold": 0.65},
    "Russia-Ukraine": {"date": "2022-03-01", "ticker": "XOM", "shock": 50, "threshold": 0.65},
}

# Synthetic cloud fallback has lower raw correlations than the full PhD dataset.
# Cap preset thresholds to keep contagion dynamics visible in demo mode.
if data_loader.is_synthetic_mode():
    CRISIS_PRESETS = {
        name: {
            **params,
            "threshold": min(float(params.get("threshold", 0.5)), 0.35),
        }
        for name, params in CRISIS_PRESETS.items()
    }

# Guided demo prompts for live presentations.
DEMO_QUERIES = {
    "1) Fast shock (local)": (
        "What happens if JPM crashes 40% on 2025-12-01? "
        "Include cascade depth and affected nodes."
    ),
    "2) Compare banks (complex)": (
        "Compare systemic risk between JPM and GS on 2025-12-01 using DebtRank, "
        "highlight differences in cascade depth and sector concentration, "
        "then propose a hedging plan for a portfolio overweight Financials and Real Estate."
    ),
    "3) Multi-stock ranking": (
        "Compare NVDA, XOM, and UNH with 50% shock on 2025-12-01. "
        "Rank by systemic impact and list top affected sectors."
    ),
    "4) Energy stress test": (
        "What happens if XOM drops 60% on 2022-03-01? Include cascade depth and sector breakdown."
    ),
    "5) Defensive allocation": (
        "Given current network regime and contagion risk, propose a conservative 30-day hedge plan."
    ),
}

BENCHMARK_QUERIES = [
    "What happens if JPM crashes 40% on 2025-12-01?",
    "Simulate AAPL 60% shock on 2024-08-05 with debtrank.",
    "Compare systemic risk between JPM and GS on 2025-12-01.",
    "Run linear threshold for NVDA 50% on 2025-12-01.",
    "Compare NVDA, XOM, and UNH with 50% shock on 2025-12-01.",
]

SCENARIO_PACK = [
    {
        "name": "A) Bank shock quick test",
        "query": "What happens if JPM crashes 40% on 2025-12-01? Include cascade depth and affected nodes.",
        "expected_route": "local_fast_mode",
    },
    {
        "name": "B) Bank comparison strategy",
        "query": (
            "Compare systemic risk between JPM and GS on 2025-12-01 using DebtRank, "
            "highlight differences in cascade depth and sector concentration, then propose a hedging plan."
        ),
        "expected_route": "gpt",
    },
    {
        "name": "C) Multi-stock systemic ranking",
        "query": "Compare NVDA, XOM, and UNH with 50% shock on 2025-12-01. Rank by systemic impact.",
        "expected_route": "gpt",
    },
    {
        "name": "D) Crisis replay",
        "query": "What happens if GS drops 50% on 2024-08-05? Include contagion waves and top vulnerable sectors.",
        "expected_route": "local_fast_mode",
    },
    {
        "name": "E) Portfolio hedging",
        "query": (
            "For a portfolio overweight Financials and Real Estate, compare shock impact for JPM and GS "
            "on 2025-12-01 and propose a conservative hedge policy."
        ),
        "expected_route": "gpt",
    },
]

STRUCTURED_SCHEMA_VERSION = "v1"

RISK_PROFILE_GUIDANCE = {
    "conservative": "Prioritize drawdown control, tighter stops, larger hedge notional, preserve liquidity.",
    "balanced": "Balance risk reduction with moderate opportunity retention.",
    "aggressive": "Accept higher volatility, use tactical hedges and selective concentration.",
}

WORKFLOW_TRANSITIONS = {
    "received": {"parsed", "blocked"},
    "parsed": {"local_ready", "gpt_ready", "blocked"},
    "local_ready": {"gpt_ready", "completed", "blocked"},
    "gpt_ready": {"gpt_running", "completed", "blocked"},
    "gpt_running": {"completed", "blocked"},
    "blocked": {"completed"},
    "completed": set(),
}

MAX_COMPARE_TICKERS = 12
CACHE_SEMANTIC_MIN_SCORE = 0.55
CIRCUIT_COOLDOWN_SEC = 90
AUTONOMOUS_SHOCK_GRID = [30, 50, 70]
DEFAULT_COMMANDER_TOP_N = 5
DEFAULT_AUTONOMOUS_SEEDS = 10
AGENTIC_OP_TIMEOUT_SEC = 25
AGENTIC_CACHE_TTL_SEC = 600
SECTOR_HEDGE_MAP = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}
PORTFOLIO_SAMPLE = "JPM,0.25\nGS,0.20\nXOM,0.15\nNVDA,0.20\nUNH,0.20"
QUERY_TOKEN_STOPWORDS = {
    "what", "happens", "happen", "if", "the", "and", "for", "with", "from",
    "then", "that", "this", "into", "using", "include", "between", "compare",
    "systemic", "risk", "current", "market", "regime", "plan", "hedging",
    "portfolio", "overweight", "underweight", "on",
}

# Company name aliases used for lightweight query parsing.
COMPANY_NAME_MAP = {
    "JPMORGAN": "JPM", "JP MORGAN": "JPM", "GOLDMAN": "GS", "GOLDMAN SACHS": "GS",
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "AMAZON": "AMZN", "TESLA": "TSLA", "NVIDIA": "NVDA", "META": "META",
    "FACEBOOK": "META", "NETFLIX": "NFLX", "BANK OF AMERICA": "BAC",
    "CITIGROUP": "C", "CITI": "C", "MORGAN STANLEY": "MS",
    "EXXON": "XOM", "CHEVRON": "CVX", "PFIZER": "PFE",
    "JOHNSON": "JNJ", "VISA": "V", "MASTERCARD": "MA",
    "DISNEY": "DIS", "BOEING": "BA", "INTEL": "INTC", "AMD": "AMD",
    "WELLS FARGO": "WFC", "BLACKROCK": "BLK",
}


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
defaults = build_app_defaults(
    is_synthetic_mode=data_loader.is_synthetic_mode(),
    demo_story=list(DEMO_QUERIES.keys())[0],
    scenario_pack_choice=SCENARIO_PACK[0]["name"],
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

AUTO_BUILD_ON_START = os.getenv("AUTO_BUILD_ON_START", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OUTLOOK_ENGINE_VERSION = "2026-03-08-live-fast-v1"

if st.session_state.get("outlook_engine_version") != OUTLOOK_ENGINE_VERSION:
    st.session_state.outlook_engine_version = OUTLOOK_ENGINE_VERSION
    st.session_state.outlook_report = None
    st.session_state.outlook_joined = None
    st.session_state.outlook_joined_by_model = None

if st.session_state.sector_dict is None or st.session_state.tickers is None:
    try:
        if st.session_state.sector_dict is None:
            st.session_state.sector_dict = data_loader.get_sector_dict()
        if st.session_state.tickers is None:
            st.session_state.tickers = data_loader.get_ticker_list()
    except Exception as exc:
        paths = data_loader.get_data_root_info()
        st.error(
            "RiskSentinel data files are not available.\n\n"
            f"- Resolved data path: `{paths['final']}`\n"
            f"- Networks path: `{paths['networks']}`\n"
            f"- RISKSENTINEL_DATA_ROOT: `{paths['env_data_root'] or '(not set)'}`\n"
            f"- Data mode: `{paths.get('data_mode', 'unknown')}`\n\n"
            "Set `RISKSENTINEL_DATA_ROOT` to a folder containing the processed dataset "
            "(including `sector_mapping.parquet` and `networks/node_centralities.pkl`).\n"
            "Or enable synthetic fallback with `RISKSENTINEL_ALLOW_SYNTHETIC_DATA=1`.\n\n"
            f"Technical error: `{type(exc).__name__}: {exc}`"
        )
        st.stop()


def _get_forecast_report_path() -> Path:
    raw = _get_runtime_value("RISKSENTINEL_FORECAST_REPORT", "")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1] / "artifacts" / "systemic_risk_forecast_latest.json"


@st.cache_data(show_spinner=False)
def _load_forecast_report_cached(path_str: str, mtime_ns: int) -> dict | None:
    del mtime_ns
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_forecast_report() -> tuple[dict | None, Path]:
    path = _get_forecast_report_path()
    if not path.is_file():
        return None, path
    report = _load_forecast_report_cached(str(path), path.stat().st_mtime_ns)
    return report, path


def run_outlook_shock_playback(
    *,
    date_str: str,
    threshold: float,
    ticker: str,
    shock_pct: int,
    model: str,
    intervention: str = "none",
) -> dict[str, object]:
    corr, actual_date = data_loader.get_correlation_matrix(date_str)
    G = network.build_network(corr, threshold=threshold, sector_dict=st.session_state.sector_dict)
    intervention_meta = _apply_policy_intervention(G, intervention)
    if ticker not in G:
        raise ValueError(f"{ticker} not in network at threshold {threshold:.2f}. Lower the threshold or choose another ticker.")
    pos = compute_layout(G)
    result = contagion.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
    return {
        "G": G,
        "pos": pos,
        "result": result,
        "date": str(actual_date.date()),
        "threshold": float(threshold),
        "ticker": ticker,
        "shock_pct": int(shock_pct),
        "model": model,
        "intervention": intervention,
        "intervention_meta": intervention_meta,
    }


def _sector_options() -> list[str]:
    return sorted({v for v in (st.session_state.sector_dict or {}).values() if v})


def _top_sector_ticker(date_str: str, sector: str) -> str:
    centralities, _ = data_loader.get_node_centralities_for_date(date_str)
    candidates = [
        (ticker, float(vals.get("pagerank", 0.0)))
        for ticker, vals in centralities.items()
        if vals.get("Sector", st.session_state.sector_dict.get(ticker)) == sector
    ]
    if not candidates:
        return st.session_state.tickers[0]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _apply_policy_intervention(G: nx.Graph, intervention: str) -> dict[str, object]:
    intervention = (intervention or "none").strip().lower()
    meta: dict[str, object] = {"policy": intervention}
    if intervention == "none":
        meta["label"] = "No intervention"
        return meta
    if G.number_of_nodes() == 0:
        meta["label"] = "No intervention"
        return meta
    if intervention == "remove_top_connector":
        centrality = network.compute_node_centralities(G)
        top = max(centrality.items(), key=lambda item: float(item[1].get("pagerank", 0.0)))[0]
        if top in G:
            G.remove_node(top)
        meta["label"] = "Remove top connector"
        meta["removed_node"] = top
        return meta
    if intervention == "cap_exposure":
        for _, _, data in G.edges(data=True):
            capped = min(float(data.get("abs_weight", 0.0)), 0.35)
            signed = float(data.get("weight", 0.0))
            data["abs_weight"] = capped
            data["weight"] = np.sign(signed) * capped
        meta["label"] = "Cap edge exposure at 0.35"
        return meta
    if intervention == "sector_firebreak":
        removed = 0
        for u, v, data in list(G.edges(data=True)):
            sector_u = G.nodes[u].get("sector") or st.session_state.sector_dict.get(u)
            sector_v = G.nodes[v].get("sector") or st.session_state.sector_dict.get(v)
            if sector_u != sector_v and float(data.get("abs_weight", 0.0)) < 0.60:
                G.remove_edge(u, v)
                removed += 1
        meta["label"] = "Sector firebreak"
        meta["removed_edges"] = removed
        return meta
    meta["label"] = "No intervention"
    return meta


def _render_evidence_model_split() -> None:
    col1, col2 = st.columns(2)
    col1.markdown(
        (
            '<div class="insight-card">'
            '<div class="insight-kicker">Deterministic Evidence</div>'
            '<div class="insight-copy">Observed correlation network, node centrality, contagion waves, affected names, cascade depth, and stress metrics.</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    col2.markdown(
        (
            '<div class="insight-card">'
            '<div class="insight-kicker">AI / Product Layer</div>'
            '<div class="insight-copy">Narrative framing, action recommendations, and policy interpretation built on top of deterministic results.</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _regime_pill_html(label: str) -> str:
    pill_class = f"pill-{label.lower()}" if label.lower() in {"calm", "normal", "elevated", "high", "crisis"} else "pill-normal"
    return f'<span class="status-pill {pill_class}">{html.escape(label)}</span>'


# ---------------------------------------------------------------------------
# NATURAL LANGUAGE PARSER (local, no LLM needed)
# ---------------------------------------------------------------------------
def extract_tickers_from_query(query: str, tickers: list[str]) -> list[str]:
    return extract_tickers_from_query_service(query, tickers, COMPANY_NAME_MAP)


def parse_chat_query(query: str) -> dict | None:
    return parse_chat_query_service(query, st.session_state.tickers, COMPANY_NAME_MAP)


def parse_structured_agent_output(text: str) -> dict | None:
    return parse_structured_agent_output_service(text, STRUCTURED_SCHEMA_VERSION)


def _get_runtime_value(name: str, default: str = "") -> str:
    return get_runtime_value_service(st, name, default)


def _get_runtime_int(name: str, default: int) -> int:
    return get_runtime_int_service(st, name, default)


@st.cache_resource
def get_global_gpt_rate_bucket() -> dict:
    """Shared in-process bucket for soft global GPT throttle."""
    return {"events": [], "day_key": "", "day_calls": 0}


def _prune_events(events: list[float], now_ts: float, window_sec: int = 60) -> list[float]:
    return prune_events_service(events, now_ts, window_sec)


def _agentic_cache_key(op_name: str, **kwargs) -> str:
    return agentic_cache_key_service(op_name, **kwargs)


def _run_agentic_operation(
    *,
    op_name: str,
    cache_key: str,
    fn: Callable[[], dict],
    timeout_sec: int = AGENTIC_OP_TIMEOUT_SEC,
    ttl_sec: int = AGENTIC_CACHE_TTL_SEC,
) -> tuple[dict, bool]:
    return run_agentic_operation_service(
        session_state=st.session_state,
        op_name=op_name,
        cache_key=cache_key,
        fn=fn,
        timeout_sec=timeout_sec,
        ttl_sec=ttl_sec,
    )


def _tokenize_query(query: str) -> set[str]:
    return tokenize_query_service(query, QUERY_TOKEN_STOPWORDS)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    return jaccard_similarity_service(a, b)


def get_gpt_access_policy() -> dict:
    return get_gpt_access_policy_service(st_module=st, session_state=st.session_state)


def unlock_judge_access(user_code: str) -> bool:
    return unlock_judge_access_service(
        st_module=st,
        session_state=st.session_state,
        user_code=user_code,
    )


def check_gpt_rate_limit() -> tuple[bool, str]:
    return check_gpt_rate_limit_service(
        session_state=st.session_state,
        get_runtime_int_fn=_get_runtime_int,
        get_global_bucket_fn=get_global_gpt_rate_bucket,
        prune_events_fn=_prune_events,
    )


def register_gpt_call() -> None:
    register_gpt_call_service(
        session_state=st.session_state,
        get_global_bucket_fn=get_global_gpt_rate_bucket,
        prune_events_fn=_prune_events,
    )


def is_gpt_circuit_open() -> tuple[bool, float]:
    return is_gpt_circuit_open_service(session_state=st.session_state)


def register_gpt_success() -> None:
    register_gpt_success_service(session_state=st.session_state)


def register_gpt_failure(reason: str) -> None:
    register_gpt_failure_service(
        session_state=st.session_state,
        reason=reason,
        cooldown_sec=CIRCUIT_COOLDOWN_SEC,
    )


def estimate_eta_seconds(history: list[dict], strategy: str, fallback: float = 18.0) -> float:
    return estimate_eta_seconds_service(history, strategy, np, fallback=fallback)


def create_run_trace(
    query: str,
    parsed: dict | None,
    complex_query: bool,
    in_scope: bool,
    scope_reason: str,
    model_for_query: str,
    threshold: float,
) -> dict:
    return create_run_trace_service(
        query=query,
        parsed=parsed,
        complex_query=complex_query,
        in_scope=in_scope,
        scope_reason=scope_reason,
        model_for_query=model_for_query,
        threshold=threshold,
    )


def trace_event(trace: dict, label: str, detail: str = "") -> None:
    trace_event_service(trace, label, detail)


def advance_workflow(trace: dict, next_state: str) -> None:
    advance_workflow_service(trace, next_state, WORKFLOW_TRANSITIONS)


def finalize_run_trace(trace: dict) -> dict:
    return finalize_run_trace_service(trace, st.session_state)


def persist_run_trace(trace: dict) -> None:
    persist_run_trace_service(trace, session_state=st.session_state, app_file=__file__)


def run_local_benchmark(threshold: float) -> dict:
    return run_local_benchmark_service(
        benchmark_queries=BENCHMARK_QUERIES,
        parse_chat_query_fn=parse_chat_query,
        infer_model_from_query_fn=infer_model_from_query,
        do_build_network_fn=do_build_network,
        contagion_module=contagion,
        snapshot_local_state_fn=snapshot_local_state,
        restore_local_state_fn=restore_local_state,
        session_state=st.session_state,
        np_module=np,
        threshold=threshold,
    )


def run_scenario_pack_eval() -> dict:
    return run_scenario_pack_eval_service(
        scenario_pack=SCENARIO_PACK,
        parse_chat_query_fn=parse_chat_query,
        is_query_in_scope_fn=is_query_in_scope,
        is_complex_query_fn=is_complex_query,
        choose_execution_policy_fn=choose_execution_policy,
        session_state=st.session_state,
        access_allowed=get_gpt_access_policy()["allowed"],
    )


def choose_execution_policy(
    *,
    parsed: dict | None,
    complex_query: bool,
    in_scope: bool,
    agent_mode: bool,
    gpt_for_parseable_queries: bool,
    access_allowed: bool,
    selected_strategy: str,
) -> dict:
    """Deterministic routing policy for local/GPT paths."""
    return agentic_ops.choose_execution_policy(
        parsed=parsed,
        complex_query=complex_query,
        in_scope=in_scope,
        agent_mode=agent_mode,
        gpt_for_parseable_queries=gpt_for_parseable_queries,
        access_allowed=access_allowed,
        selected_strategy=selected_strategy,
    )


def build_context_facts_html() -> str:
    return build_context_facts_html_service(st.session_state.graph_data)


def build_memory_hint(query: str, history: list[dict], top_k: int = 2) -> str:
    return build_memory_hint_service(query, history, top_k=top_k)


def build_structured_prompt(
    user_query: str,
    facts_plain: str,
    risk_profile: str,
    memory_hint: str = "",
    rag_context: str = "",
    evidence_gate_strict: bool = True,
) -> str:
    return build_structured_prompt_service(
        user_query=user_query,
        facts_plain=facts_plain,
        risk_profile=risk_profile,
        risk_profile_guidance=RISK_PROFILE_GUIDANCE,
        memory_hint=memory_hint,
        rag_context=rag_context,
        evidence_gate_strict=evidence_gate_strict,
    )


def build_policy_plan(
    *,
    query: str,
    parsed: dict | None,
    compare_query: bool,
    in_scope: bool,
    execution_policy: dict,
    selected_date: str,
    threshold: float,
    model_for_query: str,
) -> list[str]:
    """Deterministic planner output (Policy role) shown in explainability."""
    return agentic_ops.build_policy_plan(
        query=query,
        parsed=parsed,
        compare_query=compare_query,
        in_scope=in_scope,
        execution_policy=execution_policy,
        selected_date=selected_date,
        threshold=threshold,
        model_for_query=model_for_query,
        max_compare_tickers=MAX_COMPARE_TICKERS,
    )


def summarize_executor_log(events: list[dict], limit: int = 18) -> list[dict]:
    """Compact executor timeline from trace events."""
    return agentic_ops.summarize_executor_log(events, limit=limit)


def remember_session_decision(query: str, trace: dict) -> None:
    remember_session_decision_service(query, trace, st.session_state)


def build_session_decision_hint(query: str, top_k: int = 2) -> str:
    return build_session_decision_hint_service(
        query,
        st.session_state.session_decisions or [],
        _tokenize_query,
        _jaccard_similarity,
        top_k=top_k,
    )


def score_shock_summary(summary: dict, total_nodes: int) -> float:
    """Unified 0-100 stress score for commander/autonomous ranking."""
    return agentic_ops.score_shock_summary(summary, total_nodes)


def _build_graph_for_analysis(date_str: str, threshold: float) -> tuple[nx.Graph, str, dict, str, float]:
    return agentic_ops._build_graph_for_analysis(
        date_str=date_str,
        threshold=threshold,
        sector_dict=st.session_state.sector_dict,
        data_loader_mod=data_loader,
        network_mod=network,
    )


def run_scenario_commander(
    *,
    date_str: str,
    threshold: float,
    shock_pct: int,
    model: str,
    top_n: int = DEFAULT_COMMANDER_TOP_N,
    sector_dict: dict[str, str] | None = None,
) -> dict:
    """Agentic multi-scenario commander: picks top systemic nodes and ranks outcomes."""
    sector_data = st.session_state.get("sector_dict") if sector_dict is None else sector_dict
    return agentic_ops.run_scenario_commander(
        date_str=date_str,
        threshold=threshold,
        shock_pct=shock_pct,
        model=model,
        top_n=top_n,
        sector_dict=sector_data or {},
        data_loader_mod=data_loader,
        network_mod=network,
        contagion_mod=contagion,
    )


def run_autonomous_stress_test(
    *,
    date_str: str,
    threshold: float,
    model: str = "debtrank",
    shock_grid: list[int] | None = None,
    max_seeds: int = DEFAULT_AUTONOMOUS_SEEDS,
    sector_dict: dict[str, str] | None = None,
) -> dict:
    """Autonomous mode: explore hidden fragilities without user choosing target ticker."""
    grid = shock_grid or AUTONOMOUS_SHOCK_GRID
    sector_data = st.session_state.get("sector_dict") if sector_dict is None else sector_dict
    return agentic_ops.run_autonomous_stress_test(
        date_str=date_str,
        threshold=threshold,
        model=model,
        shock_grid=grid,
        max_seeds=max_seeds,
        sector_dict=sector_data or {},
        data_loader_mod=data_loader,
        network_mod=network,
        contagion_mod=contagion,
    )


def _parse_portfolio_positions(text: str, allowed_tickers: set[str]) -> tuple[list[dict], list[str]]:
    return agentic_ops.parse_portfolio_positions(text, allowed_tickers)


def run_portfolio_copilot(
    *,
    portfolio_text: str,
    date_str: str,
    threshold: float,
    model: str,
    stress_shock_pct: int,
    risk_profile: str | None = None,
    tickers: list[str] | None = None,
    sector_dict: dict[str, str] | None = None,
) -> dict:
    """Portfolio Co-Pilot: position-aware stress diagnostics and hedge runbook."""
    risk_profile_val = st.session_state.get("risk_profile", "balanced") if risk_profile is None else risk_profile
    ticker_list = st.session_state.get("tickers") if tickers is None else tickers
    sector_data = st.session_state.get("sector_dict") if sector_dict is None else sector_dict
    return agentic_ops.run_portfolio_copilot(
        portfolio_text=portfolio_text,
        date_str=date_str,
        threshold=threshold,
        model=model,
        stress_shock_pct=stress_shock_pct,
        risk_profile=str(risk_profile_val),
        tickers=ticker_list or [],
        sector_dict=sector_data or {},
        data_loader_mod=data_loader,
        network_mod=network,
        contagion_mod=contagion,
    )


def build_auto_portfolio_from_network(
    *,
    date_str: str,
    threshold: float,
    n_positions: int = 5,
    sector_dict: dict[str, str] | None = None,
) -> dict:
    """Generate portfolio candidates from current network (pagerank + sector diversification)."""
    sector_data = st.session_state.get("sector_dict") if sector_dict is None else sector_dict
    return agentic_ops.build_auto_portfolio_from_network(
        date_str=date_str,
        threshold=threshold,
        n_positions=n_positions,
        sector_dict=sector_data or {},
        data_loader_mod=data_loader,
        network_mod=network,
    )


def evaluate_run_trace(trace: dict) -> dict:
    return evaluate_run_trace_service(trace)


def format_llm_text_for_card(text: str) -> str:
    return format_llm_text_for_card_service(text)


def build_simulation_facts_html() -> str:
    return build_simulation_facts_html_service(
        st.session_state.graph_data,
        st.session_state.shock_result,
    )


def _compute_compare_rows(
    G: nx.Graph,
    tickers: list[str],
    shock_pct: int,
    model: str,
) -> tuple[list[dict], dict[str, contagion.ShockResult]]:
    return compute_compare_rows(
        G,
        tickers,
        shock_pct,
        model,
        sector_dict=st.session_state.sector_dict,
        contagion_module=contagion,
        max_compare_tickers=MAX_COMPARE_TICKERS,
    )


def build_compare_facts_html(
    rows: list[dict],
    date: str,
    threshold: float,
    regime: str,
    vix: float,
    shock_pct: int,
    model: str,
) -> str:
    return build_compare_facts_html_service(
        rows,
        date,
        threshold,
        regime,
        vix,
        shock_pct,
        model,
    )


def build_agent_cache_key(
    query: str,
    strategy: str,
    primary_deployment: str,
    parsed: dict | None,
    threshold: float,
    model: str,
    risk_profile: str,
    schema_version: str,
) -> str:
    return build_agent_cache_key_service(
        query=query,
        strategy=strategy,
        primary_deployment=primary_deployment,
        parsed=parsed,
        threshold=threshold,
        model=model,
        risk_profile=risk_profile,
        schema_version=schema_version,
    )


def find_cached_agent_response(
    *,
    exact_key: str,
    query: str,
    fingerprint: dict,
) -> tuple[dict | None, str]:
    return find_cached_agent_response_service(
        cache=st.session_state.agent_response_cache,
        exact_key=exact_key,
        query=query,
        fingerprint=fingerprint,
        tokenize_fn=_tokenize_query,
        similarity_fn=_jaccard_similarity,
        cache_semantic_min_score=CACHE_SEMANTIC_MIN_SCORE,
    )


def get_agent_config_status() -> tuple[bool, str]:
    return get_agent_config_status_service(get_settings)


def _run_async(coro):
    from src.ui.services.llm_ops import run_async

    return run_async(coro)


async def _run_orchestrator_query_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    from src.ui.services.llm_ops import run_orchestrator_query_async

    return await run_orchestrator_query_async(query, timeout_sec, deployment_name=deployment_name)


async def _run_simple_query_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    from src.ui.services.llm_ops import run_simple_query_async

    return await run_simple_query_async(query, timeout_sec, deployment_name=deployment_name)


async def _run_parallel_workflow_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    from src.ui.services.llm_ops import run_parallel_workflow_async

    return await run_parallel_workflow_async(query, timeout_sec, deployment_name=deployment_name)


def _run_direct_commentary_query(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    from src.ui.services.llm_ops import run_direct_commentary_query

    return run_direct_commentary_query(query, timeout_sec, deployment_name=deployment_name)


async def _run_critic_validation_async(
    *,
    query: str,
    facts_plain: str,
    candidate_json_text: str,
    timeout_sec: int = 12,
    deployment_name: str | None = None,
) -> dict:
    from src.ui.services.llm_ops import run_critic_validation_async

    return await run_critic_validation_async(
        query=query,
        facts_plain=facts_plain,
        candidate_json_text=candidate_json_text,
        timeout_sec=timeout_sec,
        deployment_name=deployment_name,
        extract_json_payload_fn=extract_json_payload,
    )


def run_critic_validation(
    *,
    query: str,
    facts_plain: str,
    candidate_json_text: str,
    timeout_sec: int = 12,
    deployment_name: str | None = None,
) -> dict:
    return run_critic_validation_service(
        query=query,
        facts_plain=facts_plain,
        candidate_json_text=candidate_json_text,
        timeout_sec=timeout_sec,
        deployment_name=deployment_name,
        extract_json_payload_fn=extract_json_payload,
    )


def run_agent_query(
    query: str,
    timeout_sec: int = 35,
    strategy: str = "simple",
    deployment_name: str | None = None,
) -> str:
    return run_agent_query_service(
        query=query,
        timeout_sec=timeout_sec,
        strategy=strategy,
        deployment_name=deployment_name,
    )


def is_rate_limit_error(exc: Exception) -> bool:
    return is_rate_limit_error_service(exc)


def is_timeout_error(exc: Exception) -> bool:
    return is_timeout_error_service(exc)


def is_retryable_gpt_error(exc: Exception) -> bool:
    return is_retryable_gpt_error_service(exc)


def run_agent_query_with_backoff(
    query: str,
    timeout_sec: int,
    strategy: str,
    deployment_name: str | None = None,
    max_retries: int = 2,
    base_delay_sec: float = 1.5,
    max_total_wait_sec: float = 24.0,
    on_backoff: Callable[[float, int, int], None] | None = None,
) -> str:
    return run_agent_query_with_backoff_service(
        query=query,
        timeout_sec=timeout_sec,
        strategy=strategy,
        deployment_name=deployment_name,
        max_retries=max_retries,
        base_delay_sec=base_delay_sec,
        max_total_wait_sec=max_total_wait_sec,
        on_backoff=on_backoff,
    )


def get_deployment_routing(high_quality_mode: bool) -> tuple[str, str]:
    return get_deployment_routing_service(
        high_quality_mode=high_quality_mode,
        get_runtime_value_fn=_get_runtime_value,
    )


def run_gpt_diagnostic() -> str:
    return run_gpt_diagnostic_service(run_agent_query)


# ---------------------------------------------------------------------------
# CORE: BUILD NETWORK
# ---------------------------------------------------------------------------
def do_build_network(date_str: str, threshold: float, emit_messages: bool = True):
    build = execute_build_network(
        data_loader_obj=data_loader,
        network_module=network,
        compute_layout_fn=compute_layout,
        sector_dict=st.session_state.sector_dict,
        date_str=date_str,
        threshold=threshold,
    )
    G = build["G"]
    st.session_state.pos = build["pos"]
    st.session_state.graph_data = build["graph_data"]
    st.session_state.shock_result = None
    st.session_state.current_wave = -1

    if emit_messages:
        st.session_state.agent_messages.append(
            ("Architect", "🔧", "agent-architect", str(build["architect_message"]))
        )
    return G


# ---------------------------------------------------------------------------
# CORE: RUN SHOCK
# ---------------------------------------------------------------------------
def do_run_shock(G: nx.Graph, ticker: str, shock_pct: int, model: str, emit_messages: bool = True):
    shock = execute_shock_scenario(
        G=G,
        ticker=ticker,
        shock_pct=shock_pct,
        model=model,
        sector_dict=st.session_state.sector_dict,
        risk_profile=st.session_state.get("risk_profile", "balanced"),
        network_module=network,
        contagion_module=contagion,
    )
    if not shock.get("ok"):
        if emit_messages:
            st.session_state.agent_messages.extend(shock.get("messages", []))
        return

    st.session_state.shock_result = shock["result"]
    st.session_state.current_wave = int(shock["current_wave"])
    if emit_messages:
        st.session_state.agent_messages.extend(shock.get("messages", []))


def agent_message(name: str, icon: str, css_class: str, text: str):
    """Render a styled agent message."""
    st.markdown(
        f'<div class="agent-msg {css_class}">'
        f'<b>{icon} {name}</b><br>{text}</div>',
        unsafe_allow_html=True,
    )


sidebar_state = render_sidebar(
    {
        "st": st,
        "pd": pd,
        "data_loader": data_loader,
        "DEMO_QUERIES": DEMO_QUERIES,
        "SCENARIO_PACK": SCENARIO_PACK,
        "CRISIS_PRESETS": CRISIS_PRESETS,
        "PORTFOLIO_SAMPLE": PORTFOLIO_SAMPLE,
    }
)
selected_date = str(sidebar_state["selected_date"])
shocked_ticker = str(sidebar_state["shocked_ticker"])
shock_pct = int(sidebar_state["shock_pct"])
shock_model = str(sidebar_state["shock_model"])
threshold = float(sidebar_state["threshold"])
st.session_state.sel_date = selected_date
st.session_state.sel_ticker = shocked_ticker
st.session_state.sel_shock = shock_pct
st.session_state.sel_model = shock_model
st.session_state.sel_threshold = threshold
build_btn = bool(sidebar_state["build_btn"])
shock_btn = bool(sidebar_state["shock_btn"])
compare_btn = bool(sidebar_state["compare_btn"])
commander_btn = bool(sidebar_state["commander_btn"])
autonomous_btn = bool(sidebar_state["autonomous_btn"])
auto_portfolio_btn = bool(sidebar_state["auto_portfolio_btn"])
portfolio_btn = bool(sidebar_state["portfolio_btn"])
full_demo_btn = bool(sidebar_state["full_demo_btn"])
preset_triggered = sidebar_state.get("preset_triggered")
if preset_triggered:
    handle_preset_trigger(
        st.session_state,
        preset_triggered["params"],
        do_build_network,
        do_run_shock,
    )
    st.rerun()


# ---------------------------------------------------------------------------
# BUILD / SHOCK ACTIONS
# ---------------------------------------------------------------------------
sector_dict_ctx = dict(st.session_state.get("sector_dict") or {})
tickers_ctx = list(st.session_state.get("tickers") or [])
risk_profile_ctx = str(st.session_state.get("risk_profile", "balanced"))
auto_portfolio_n_ctx = int(st.session_state.get("auto_portfolio_n", 5) or 5)
portfolio_text_ctx = str(st.session_state.get("portfolio_text", "") or "")
agentic_requested = bool(commander_btn or autonomous_btn or auto_portfolio_btn or portfolio_btn or full_demo_btn)
sector_dict_ctx, tickers_ctx, agentic_ctx_error = ensure_agentic_context(
    st.session_state,
    data_loader,
    agentic_requested,
)
if agentic_ctx_error:
    st.session_state.last_agentic_action = agentic_ctx_error

if build_btn or (AUTO_BUILD_ON_START and st.session_state.graph_data is None):
    run_build_action(st.session_state, selected_date, threshold, do_build_network)

if shock_btn and st.session_state.graph_data:
    run_shock_action(st.session_state, shocked_ticker, shock_pct, shock_model, do_run_shock)

if compare_btn and st.session_state.graph_data:
    run_compare_action(st.session_state, shocked_ticker, shock_pct, contagion)

run_sidebar_agentic_actions(
    st_module=st,
    session_state=st.session_state,
    selected_date=selected_date,
    threshold=threshold,
    shock_pct=shock_pct,
    shock_model=shock_model,
    sector_dict_ctx=sector_dict_ctx,
    tickers_ctx=tickers_ctx,
    risk_profile_ctx=risk_profile_ctx,
    auto_portfolio_n_ctx=auto_portfolio_n_ctx,
    portfolio_text_ctx=portfolio_text_ctx,
    commander_btn=commander_btn,
    autonomous_btn=autonomous_btn,
    auto_portfolio_btn=auto_portfolio_btn,
    portfolio_btn=portfolio_btn,
    full_demo_btn=full_demo_btn,
    default_commander_top_n=DEFAULT_COMMANDER_TOP_N,
    autonomous_shock_grid=AUTONOMOUS_SHOCK_GRID,
    default_autonomous_seeds=DEFAULT_AUTONOMOUS_SEEDS,
    cache_key_fn=_agentic_cache_key,
    run_agentic_operation_fn=_run_agentic_operation,
    run_scenario_commander_fn=run_scenario_commander,
    run_autonomous_stress_test_fn=run_autonomous_stress_test,
    build_auto_portfolio_from_network_fn=build_auto_portfolio_from_network,
    run_portfolio_copilot_fn=run_portfolio_copilot,
    do_build_network_fn=do_build_network,
    agentic_ops_module=agentic_ops,
)


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    '<h1 style="color:white; margin:0;">🛡️ RiskSentinel</h1>'
    f'<p style="color:{PALETTE["text_muted"]}; margin:0;">Systemic Surveillance and Forward Stress Testing for Financial Contagion</p>'
    '</div>',
    unsafe_allow_html=True,
)


def _build_outlook_tab_context() -> dict[str, object]:
    effective_outlook_scenarios = OUTLOOK_SCENARIOS
    if data_loader.is_synthetic_mode():
        effective_outlook_scenarios = [
            {
                **scenario,
                "threshold": min(float(scenario.get("threshold", 0.5)), 0.35),
            }
            for scenario in OUTLOOK_SCENARIOS
        ]
    return {
        "st": st,
        "pd": pd,
        "np": np,
        "html": html,
        "json": json,
        "data_loader": data_loader,
        "forecast_nearest_leq": forecast_nearest_leq,
        "_get_forecast_date_bounds": get_forecast_date_bounds,
        "_run_live_outlook_cached": run_live_outlook_cached,
        "load_forecast_report": load_forecast_report,
        "_get_forecast_report_path": _get_forecast_report_path,
        "_summary_rows_from_forecast": summary_rows_from_forecast,
        "_compute_outlook_snapshot": compute_outlook_snapshot,
        "_build_change_rows": build_change_rows,
        "_regime_pill_html": _regime_pill_html,
        "_build_regime_transition_copy": build_regime_transition_copy,
        "_forecast_confidence_copy": forecast_confidence_copy,
        "_format_outlook_metric_label": format_outlook_metric_label,
        "build_outlook_compact_figure": lambda joined, metric, focus_date=None: build_outlook_compact_figure_chart(
            joined,
            metric,
            palette=PALETTE,
            focus_date=focus_date,
        ),
        "build_outlook_spread_figure": lambda joined, metric, focus_date=None: build_outlook_spread_figure_chart(
            joined,
            metric,
            palette=PALETTE,
            focus_date=focus_date,
        ),
        "_build_outlook_checkpoint_rows": build_outlook_checkpoint_rows,
        "build_outlook_animation_figure": lambda joined, metric, focus_date=None: build_outlook_animation_figure_chart(
            joined,
            metric,
            palette=PALETTE,
            focus_date=focus_date,
        ),
        "build_outlook_timeseries_figure": lambda joined, metric, focus_date=None: build_outlook_timeseries_figure_chart(
            joined,
            metric,
            palette=PALETTE,
            focus_date=focus_date,
        ),
        "_build_narrative_lines": build_narrative_lines,
        "_top_systemic_rows": lambda date_str, limit=5: top_systemic_rows(
            data_loader,
            st.session_state.sector_dict,
            date_str,
            limit=limit,
        ),
        "_build_action_rows": build_action_rows,
        "_build_why_this_matters_rows": lambda _snapshot: build_why_this_matters_rows(),
        "_build_watchlist_rows": lambda focus_date, lookback=20, limit=8: build_watchlist_rows(
            data_loader,
            st.session_state.sector_dict,
            focus_date,
            lookback=lookback,
            limit=limit,
        ),
        "OUTLOOK_SCENARIOS": effective_outlook_scenarios,
        "run_outlook_shock_playback": run_outlook_shock_playback,
        "_sector_options": _sector_options,
        "_top_sector_ticker": _top_sector_ticker,
        "_build_compare_rows": build_compare_rows,
        "_build_counterfactual_row": build_counterfactual_row,
        "_build_vulnerability_rows": lambda date_str, shock_bundle, limit=8: build_vulnerability_rows(
            data_loader,
            st.session_state.sector_dict,
            date_str,
            shock_bundle,
            limit=limit,
        ),
        "_build_why_nodes_rows": lambda date_str, shock_bundle, limit=6: build_why_nodes_rows(
            data_loader,
            st.session_state.sector_dict,
            date_str,
            shock_bundle,
            limit=limit,
        ),
        "_render_evidence_model_split": _render_evidence_model_split,
        "build_animated_figure": lambda G, pos, result, blast_radius_only=False: build_animated_figure_chart(
            G,
            pos,
            result,
            sector_dict=st.session_state.sector_dict,
            sector_colors=data_loader.SECTOR_COLORS,
            risk_colors=RISK_COLORS,
            palette=PALETTE,
            edge_bg_color=PLOT_EDGE_BG,
            edge_stress_color=PLOT_EDGE_STRESS,
            blast_radius_only=blast_radius_only,
        ),
    }


def _build_surveillance_tab_context() -> dict[str, object]:
    return {
        "st": st,
        "pd": pd,
        "go": go,
        "MODEL_COLORS": MODEL_COLORS,
        "RISK_COLORS": RISK_COLORS,
        "PALETTE": PALETTE,
        "ui_panels": ui_panels,
        "build_systemic_risk_gauge_figure": lambda result, total_nodes: build_systemic_risk_gauge_figure_chart(
            result,
            total_nodes,
            palette=PALETTE,
        ),
        "build_sector_impact_bar_figure": lambda result, top_n=10: build_sector_impact_bar_figure_chart(
            result,
            sector_dict=st.session_state.sector_dict,
            palette=PALETTE,
            stress_colorscale=PLOTLY_STRESS_COLORSCALE,
            top_n=top_n,
        ),
        "build_stress_tier_donut_figure": lambda result: build_stress_tier_donut_figure_chart(
            result,
            palette=PALETTE,
            risk_colors=RISK_COLORS,
        ),
        "build_wave_trend_figure": lambda result: build_wave_trend_figure_chart(
            result,
            palette=PALETTE,
        ),
        "build_timeline_figure": lambda: build_timeline_figure_chart(
            data_loader.load_network_metrics(),
            palette=PALETTE,
            crisis_events=data_loader.CRISIS_EVENTS,
            selected_date=(st.session_state.graph_data or {}).get("date"),
            event_fill=PLOT_EVENT_FILL,
        ),
        "build_severity_df": lambda result: build_severity_df(result, st.session_state.sector_dict),
    }


def _build_audit_trail_tab_context() -> dict[str, object]:
    return {
        "st": st,
        "pd": pd,
        "np": np,
        "ui_panels": ui_panels,
        "summarize_quality": summarize_quality,
        "build_judge_kpis": build_judge_kpis,
        "build_judge_run_rows": build_judge_run_rows,
        "SCENARIO_PACK": SCENARIO_PACK,
        "generate_report_text": lambda: generate_report_text(
            graph_data=st.session_state.graph_data,
            shock_result=st.session_state.shock_result,
            sector_dict=st.session_state.sector_dict,
            agent_messages=st.session_state.agent_messages,
        ),
        "generate_report_markdown": lambda: generate_report_markdown(
            graph_data=st.session_state.graph_data,
            shock_result=st.session_state.shock_result,
            last_run_metrics=st.session_state.last_run_metrics,
        ),
        "generate_trace_bundle_json": lambda: generate_trace_bundle_json(
            last_run_metrics=st.session_state.last_run_metrics,
            run_trace=st.session_state.run_trace,
            run_trace_history=st.session_state.run_trace_history,
            rag_last_docs=st.session_state.rag_last_docs,
            risk_profile=st.session_state.risk_profile,
            latest_policy_plan=st.session_state.latest_policy_plan,
            latest_executor_log=st.session_state.latest_executor_log,
            session_decisions=st.session_state.session_decisions,
            commander_results=st.session_state.commander_results,
            autonomous_results=st.session_state.autonomous_results,
            portfolio_copilot=st.session_state.portfolio_copilot,
        ),
        "build_submission_bundle_bytes": lambda: build_submission_bundle_bytes(
            report_text=generate_report_text(
                graph_data=st.session_state.graph_data,
                shock_result=st.session_state.shock_result,
                sector_dict=st.session_state.sector_dict,
                agent_messages=st.session_state.agent_messages,
            ),
            brief_markdown=generate_report_markdown(
                graph_data=st.session_state.graph_data,
                shock_result=st.session_state.shock_result,
                last_run_metrics=st.session_state.last_run_metrics,
            ),
            action_ceo_brief=generate_action_pack_ceo_brief(
                graph_data=st.session_state.graph_data,
                shock_result=st.session_state.shock_result,
                commander=st.session_state.commander_results,
                autonomous=st.session_state.autonomous_results,
                portfolio=st.session_state.portfolio_copilot,
            ),
            action_runbook=generate_action_pack_runbook(
                commander=st.session_state.commander_results,
                portfolio=st.session_state.portfolio_copilot,
            ),
            action_machine_json=generate_action_pack_machine_json(
                graph_data=st.session_state.graph_data,
                commander=st.session_state.commander_results,
                autonomous_stress_test=st.session_state.autonomous_results,
                portfolio_copilot=st.session_state.portfolio_copilot,
                trace_summary=st.session_state.run_trace,
                policy_plan=st.session_state.latest_policy_plan,
                executor_log=st.session_state.latest_executor_log,
                session_memory=st.session_state.session_decisions[-20:],
            ),
            trace_json=generate_trace_bundle_json(
                last_run_metrics=st.session_state.last_run_metrics,
                run_trace=st.session_state.run_trace,
                run_trace_history=st.session_state.run_trace_history,
                rag_last_docs=st.session_state.rag_last_docs,
                risk_profile=st.session_state.risk_profile,
                latest_policy_plan=st.session_state.latest_policy_plan,
                latest_executor_log=st.session_state.latest_executor_log,
                session_decisions=st.session_state.session_decisions,
                commander_results=st.session_state.commander_results,
                autonomous_results=st.session_state.autonomous_results,
                portfolio_copilot=st.session_state.portfolio_copilot,
            ),
            run_trace_history=st.session_state.run_trace_history,
            scenario_eval_results=st.session_state.scenario_eval_results,
            rag_last_docs=st.session_state.rag_last_docs,
        ),
        "generate_action_pack_ceo_brief": lambda: generate_action_pack_ceo_brief(
            graph_data=st.session_state.graph_data,
            shock_result=st.session_state.shock_result,
            commander=st.session_state.commander_results,
            autonomous=st.session_state.autonomous_results,
            portfolio=st.session_state.portfolio_copilot,
        ),
        "generate_action_pack_runbook": lambda: generate_action_pack_runbook(
            commander=st.session_state.commander_results,
            portfolio=st.session_state.portfolio_copilot,
        ),
        "generate_action_pack_machine_json": lambda: generate_action_pack_machine_json(
            graph_data=st.session_state.graph_data,
            commander=st.session_state.commander_results,
            autonomous_stress_test=st.session_state.autonomous_results,
            portfolio_copilot=st.session_state.portfolio_copilot,
            trace_summary=st.session_state.run_trace,
            policy_plan=st.session_state.latest_policy_plan,
            executor_log=st.session_state.latest_executor_log,
            session_memory=st.session_state.session_decisions[-20:],
        ),
        "selected_date": selected_date,
    }


def _build_stress_lab_tab_context() -> dict[str, object]:
    return {
        "st": st,
        "build_animated_figure": lambda G, pos, result, blast_radius_only=False: build_animated_figure_chart(
            G,
            pos,
            result,
            sector_dict=st.session_state.sector_dict,
            sector_colors=data_loader.SECTOR_COLORS,
            risk_colors=RISK_COLORS,
            palette=PALETTE,
            edge_bg_color=PLOT_EDGE_BG,
            edge_stress_color=PLOT_EDGE_STRESS,
            blast_radius_only=blast_radius_only,
        ),
        "build_graph_figure": lambda G, pos: build_graph_figure_chart(
            G,
            pos,
            sector_dict=st.session_state.sector_dict,
            sector_colors=data_loader.SECTOR_COLORS,
            risk_colors=RISK_COLORS,
            palette=PALETTE,
            edge_bg_color=PLOT_EDGE_BG,
        ),
        "agent_message": agent_message,
        "build_compare_rows_df": build_compare_rows_df,
        "MAX_COMPARE_TICKERS": MAX_COMPARE_TICKERS,
    }

tab_simulate, tab_dashboard, tab_outlook, tab_explain, tab_settings = st.tabs(
    ["🌐 Stress Lab", "📊 Surveillance", "🔮 Outlook", "🔍 Audit Trail", "⚙️ Ops"]
)
if st.session_state.last_agentic_action:
    st.caption(f"Agentic Ops: {st.session_state.last_agentic_action}")
if (
    st.session_state.commander_results
    or st.session_state.autonomous_results
    or st.session_state.portfolio_copilot
    or st.session_state.full_demo_last_run
):
    with st.expander("Latest Agentic Ops Result", expanded=True):
        if st.session_state.full_demo_last_run:
            demo = st.session_state.full_demo_last_run
            st.markdown(f"**Full Demo:** status `{demo.get('status', 'n/a')}`")
            steps = demo.get("steps", [])
            if steps:
                st.dataframe(pd.DataFrame(steps), use_container_width=True, hide_index=True)
        if st.session_state.commander_results and st.session_state.commander_results.get("top_pick"):
            top = st.session_state.commander_results["top_pick"]
            st.markdown(
                f"**Commander:** {top['ticker']} ({top['sector']}) | "
                f"score {top['risk_score']:.1f} | depth {top['cascade_depth']}"
            )
        if st.session_state.autonomous_results and st.session_state.autonomous_results.get("rows"):
            lead = st.session_state.autonomous_results["rows"][0]
            st.markdown(
                f"**Autonomous:** {lead['ticker']} @ {lead['shock_pct']}% | "
                f"score {lead['risk_score']:.1f} | affected {lead['n_affected']}"
            )
        if st.session_state.portfolio_copilot:
            cop = st.session_state.portfolio_copilot
            if cop.get("ok"):
                st.markdown(
                    f"**Portfolio:** stress {cop.get('expected_stress_pct', 0.0):.1f}% "
                    f"→ {cop.get('expected_stress_pct_after_hedge', 0.0):.1f}% "
                    f"(avoided ~{cop.get('estimated_loss_avoided_pct', 0.0):.1f}%)"
                )
                formula_md = ui_panels.business_kpi_formula_markdown(cop.get("kpi"))
                if formula_md:
                    st.markdown(formula_md)
            else:
                errs = (cop.get("errors") or [])[:2]
                st.warning("Portfolio input non valido. " + (" | ".join(errs) if errs else ""))
        if st.session_state.portfolio_text.strip():
            st.markdown("**Portfolio input corrente**")
            st.code(st.session_state.portfolio_text.strip(), language="text")
        st.caption("Dettagli completi nel tab Surveillance.")

# === CHAT INPUT ===
raw_chat_query = st.chat_input("Ask RiskSentinel... (e.g. 'What if Tesla crashes 60%?')")
pending_chat_query = (st.session_state.pending_chat_query or "").strip()
incoming_query = raw_chat_query if raw_chat_query else (pending_chat_query if pending_chat_query else None)
if incoming_query:
    if pending_chat_query and not raw_chat_query:
        st.session_state.pending_chat_query = ""

    chat_query = normalize_chat_query(incoming_query)
    st.session_state.chat_history.append(("user", chat_query))
    parsed = parse_chat_query(chat_query)
    compare_query = is_compare_query(chat_query, parsed)
    complex_query = is_complex_query(chat_query)
    in_scope, scope_reason = is_query_in_scope(chat_query, parsed)
    model_for_query = infer_model_from_query(chat_query)
    runtime_access_policy = get_gpt_access_policy()
    primary_deployment, fallback_deployment = get_deployment_routing(st.session_state.high_quality_mode)
    st.session_state.agent_messages = []
    st.session_state.last_structured_payload = None
    st.session_state.compare_rows_local = []
    st.session_state.compare_meta = {}
    st.session_state.rag_last_docs = []
    st.session_state.run_cancel_requested = False
    t_start = time.perf_counter()
    local_sec = None
    gpt_sec = None
    gpt_attempted = False
    gpt_success = False
    engine_label = "Local engine"
    run_state = "local_only"
    trace = create_run_trace(
        query=chat_query,
        parsed=parsed,
        complex_query=complex_query,
        in_scope=in_scope,
        scope_reason=scope_reason,
        model_for_query=model_for_query,
        threshold=threshold,
    )
    trace["_t0"] = t_start
    trace["policy"] = {
        "agent_mode": st.session_state.agent_mode,
        "strategy": st.session_state.agent_strategy,
        "high_quality_mode": st.session_state.high_quality_mode,
        "gpt_for_parseable_queries": st.session_state.gpt_for_parseable_queries,
        "evidence_gate_strict": st.session_state.evidence_gate_strict,
        "risk_profile": st.session_state.risk_profile,
        "tool_contract_version": "mcp.tool.result.v1",
        "primary_deployment": primary_deployment,
        "fallback_deployment": fallback_deployment,
        "gpt_access_allowed": runtime_access_policy["allowed"],
        "gpt_access_reason": runtime_access_policy["reason"],
    }
    execution_policy = choose_execution_policy(
        parsed=parsed,
        complex_query=complex_query,
        in_scope=in_scope,
        agent_mode=st.session_state.agent_mode,
        gpt_for_parseable_queries=st.session_state.gpt_for_parseable_queries,
        access_allowed=runtime_access_policy["allowed"],
        selected_strategy=st.session_state.agent_strategy,
    )
    planned_steps = build_policy_plan(
        query=chat_query,
        parsed=parsed,
        compare_query=compare_query,
        in_scope=in_scope,
        execution_policy=execution_policy,
        selected_date=selected_date,
        threshold=threshold,
        model_for_query=model_for_query,
    )
    trace["policy"]["router"] = execution_policy
    trace["policy"]["planned_steps"] = planned_steps
    st.session_state.latest_policy_plan = planned_steps
    trace_event(trace, "query_received", chat_query)
    trace_event(trace, "scope_check", scope_reason)
    trace_event(trace, "router_decision", execution_policy["route"])
    advance_workflow(trace, "parsed")

    with st.status("Processing query...", expanded=True) as status:
        progress = st.progress(5)
        phase = st.empty()
        elapsed_box = st.empty()
        eta_box = st.empty()
        cancel_box = st.empty()
        cancel_key = f"cancel_run_{trace['id']}"
        if cancel_box.button("Cancel pending GPT steps", key=cancel_key):
            st.session_state.run_cancel_requested = True
            trace_event(trace, "cancel_requested", "user clicked cancel")

        def _step(pct: int, text: str) -> None:
            phase.markdown(f"**{text}**")
            progress.progress(pct)
            elapsed_box.caption(f"Elapsed: {time.perf_counter() - t_start:.1f}s")
            if st.session_state.run_cancel_requested:
                eta_box.caption("Cancel requested: run will stop at next safe checkpoint.")

        def _heartbeat(phase_text: str, eta_sec: float) -> None:
            elapsed_now = time.perf_counter() - t_start
            phase.markdown(f"**{phase_text}**")
            elapsed_box.caption(f"Elapsed: {elapsed_now:.1f}s")
            eta_box.caption(f"ETA: {max(0.0, eta_sec):.1f}s")

        def _run_with_heartbeat(callable_fn, phase_text: str, eta_sec: float):
            # Run GPT calls on the main Streamlit script thread. The Agent
            # Framework / Azure stack can touch context that is not safe to
            # access from a worker thread, which surfaces as NoSessionContext.
            _heartbeat(phase_text, eta_sec)
            if st.session_state.run_cancel_requested:
                raise RuntimeError("RunCancelledByUser")
            return callable_fn()

        _step(10, "Parsing user input")
        trace_event(
            trace,
            "parse_complete",
            f"parsed={bool(parsed)}, complex={complex_query}, compare={compare_query}",
        )

        if not in_scope:
            _step(35, "Out-of-scope query for this app")
            st.session_state.agent_messages.append(
                (
                    "Sentinel",
                    "🛡️",
                    "agent-sentinel",
                    "Scope guardrail active. Ask about <b>network topology</b>, <b>crisis regime</b>, "
                    "<b>contagion simulation</b>, or <b>hedging tied to simulation outputs</b>. "
                    "Example: <i>What happens if NVDA crashes 50% on 2025-12-01?</i>",
                )
            )
            run_state = "out_of_scope"
            trace_event(trace, "guardrail_block", "query rejected by domain policy")
            advance_workflow(trace, "blocked")

        else:
            # Local deterministic path (used both for fast answers and deterministic facts anchoring).
            run_local_first = bool(execution_policy.get("run_local_first", bool(parsed)))
            suppress_local_messages = bool(complex_query and st.session_state.agent_mode)
            if run_local_first:
                advance_workflow(trace, "local_ready")
                _step(30, "Running local network build and contagion simulation")
                local_target = ",".join(parsed.get("tickers", [])) if compare_query else parsed.get("ticker", "n/a")
                trace_event(trace, "local_start", f"target={local_target}, shock={parsed['shock']}")
                t_local = time.perf_counter()
                date = parsed.get("date") or selected_date
                if not suppress_local_messages:
                    target_text = (
                        ", ".join(parsed.get("tickers", [])[:MAX_COMPARE_TICKERS])
                        if compare_query else parsed["ticker"]
                    )
                    st.session_state.agent_messages.append(
                        ("Sentinel", "🛡️", "agent-sentinel",
                         f'Understanding query: "<i>{chat_query}</i>"<br>'
                         f'→ Target: <b>{target_text}</b>, Shock: <b>{parsed["shock"]}%</b>')
                    )
                G = do_build_network(date, threshold, emit_messages=not suppress_local_messages)
                if compare_query:
                    compare_rows, result_by_ticker = _compute_compare_rows(
                        G=G,
                        tickers=parsed.get("tickers", []),
                        shock_pct=parsed["shock"],
                        model=model_for_query,
                    )
                    st.session_state.compare_rows_local = compare_rows
                    st.session_state.compare_meta = {
                        "requested_tickers": parsed.get("tickers", []),
                        "evaluated_tickers": [r["ticker"] for r in compare_rows],
                        "max_tickers": MAX_COMPARE_TICKERS,
                    }
                    if compare_rows:
                        top_ticker = compare_rows[0]["ticker"]
                        st.session_state.shock_result = result_by_ticker.get(top_ticker)
                        if st.session_state.shock_result:
                            st.session_state.current_wave = st.session_state.shock_result.cascade_depth
                        if not suppress_local_messages:
                            st.session_state.agent_messages.append(
                                (
                                    "Quant",
                                    "📊",
                                    "agent-quant",
                                    f"<b>Deterministic Compare</b> — evaluated "
                                    f"<b>{len(compare_rows)}</b> tickers ({model_for_query}) at {parsed['shock']}% shock.<br>"
                                    f"Top impact: <b>{top_ticker}</b> (total stress {compare_rows[0]['total_stress']:.2f}).",
                                )
                            )
                    else:
                        st.session_state.agent_messages.append(
                            (
                                "Sentinel",
                                "🛡️",
                                "agent-sentinel",
                                "No valid tickers found in graph for compare analysis.",
                            )
                        )
                else:
                    do_run_shock(
                        G,
                        parsed["ticker"],
                        parsed["shock"],
                        model_for_query,
                        emit_messages=not suppress_local_messages,
                    )
                local_sec = time.perf_counter() - t_local
                trace_event(trace, "local_done", f"{local_sec:.3f}s")

            if parsed and complex_query and st.session_state.agent_mode:
                st.session_state.agent_messages.append(
                    ("Sentinel", "🛡️", "agent-sentinel",
                     "Complex intent detected (comparison/strategy). Prioritizing GPT analysis.")
                )
                trace_event(trace, "routing_hint", "complex_intent=true")

            should_run_gpt = bool(execution_policy.get("should_run_gpt"))
            gpt_policy_block_reason = ""
            circuit_open, circuit_wait = is_gpt_circuit_open()
            if not runtime_access_policy["allowed"]:
                gpt_policy_block_reason = f"access_locked:{runtime_access_policy['reason']}"
            elif should_run_gpt and circuit_open:
                should_run_gpt = False
                gpt_policy_block_reason = f"circuit_open:{circuit_wait:.0f}s"
                trace_event(trace, "gpt_policy_block", gpt_policy_block_reason)
                st.session_state.agent_messages.append(
                    (
                        "Sentinel",
                        "🛡️",
                        "agent-sentinel",
                        f"GPT circuit breaker active. Retrying in ~{circuit_wait:.0f}s. Using local output.",
                    )
                )
            if should_run_gpt:
                rate_ok, rate_reason = check_gpt_rate_limit()
                if not rate_ok:
                    should_run_gpt = False
                    gpt_policy_block_reason = rate_reason
                    st.session_state.gpt_rate_limit_hits += 1
                    trace_event(trace, "gpt_policy_block", rate_reason)
                    st.session_state.agent_messages.append(
                        (
                            "Sentinel",
                            "🛡️",
                            "agent-sentinel",
                            "GPT temporarily throttled by policy. Using local simulation output only.",
                        )
                    )
                    if parsed:
                        run_state = "gpt_policy_block_local"
            trace["policy"]["should_run_gpt"] = should_run_gpt
            trace["policy"]["gpt_block_reason"] = gpt_policy_block_reason or "none"

            # Optional GPT path.
            if should_run_gpt:
                advance_workflow(trace, "gpt_ready")
                gpt_attempted = True
                t_gpt = time.perf_counter()
                strategy = execution_policy.get("effective_strategy", st.session_state.agent_strategy)
                timeout_policy = int(execution_policy.get("timeout_sec", st.session_state.agent_timeout_sec))
                retry_policy = int(execution_policy.get("max_retries", 1))
                timeout_sec_effective = min(timeout_policy, st.session_state.agent_timeout_sec)
                cache_key = build_agent_cache_key(
                    query=chat_query,
                    strategy=strategy,
                    primary_deployment=primary_deployment,
                    parsed=parsed,
                    threshold=threshold,
                    model=model_for_query,
                    risk_profile=st.session_state.risk_profile,
                    schema_version=STRUCTURED_SCHEMA_VERSION,
                )
                fingerprint = build_cache_fingerprint(
                    parsed=parsed,
                    threshold=threshold,
                    model=model_for_query,
                    risk_profile=st.session_state.risk_profile,
                    schema_version=STRUCTURED_SCHEMA_VERSION,
                    strategy=strategy,
                )

                facts_html = build_simulation_facts_html() if parsed else build_context_facts_html()
                facts_mode = "single" if parsed else "context"
                if compare_query and st.session_state.graph_data:
                    gd = st.session_state.graph_data
                    compare_rows = st.session_state.compare_rows_local
                    if not compare_rows:
                        compare_rows, _ = _compute_compare_rows(
                            G=gd["G"],
                            tickers=parsed.get("tickers", []),
                            shock_pct=parsed["shock"],
                            model=model_for_query,
                        )
                        st.session_state.compare_rows_local = compare_rows
                    facts_html = build_compare_facts_html(
                        rows=compare_rows,
                        date=str(gd.get("date", parsed.get("date") or "n/a")),
                        threshold=float(gd.get("threshold", threshold)),
                        regime=str(gd.get("regime", "n/a")),
                        vix=float(gd.get("vix", 0.0)),
                        shock_pct=parsed["shock"],
                        model=model_for_query,
                    )
                    facts_mode = "compare"
                if not facts_html:
                    facts_mode = "none"

                trace["policy"]["facts_mode"] = facts_mode
                trace_event(trace, "gpt_start", f"strategy={strategy}, facts={facts_mode}")
                advance_workflow(trace, "gpt_running")

                if facts_mode == "none":
                    facts_plain = "No deterministic facts available in local engine state."
                    trace_event(trace, "evidence_warning", "facts missing; request must return insufficient_data")
                else:
                    facts_plain = re.sub(r"<[^>]+>", "", facts_html).replace("&nbsp;", " ")
                    trace["policy"]["facts_preview"] = facts_plain[:500]

                memory_hint_parts: list[str] = []
                if st.session_state.use_session_memory:
                    hist_hint = build_memory_hint(chat_query, st.session_state.run_trace_history)
                    if hist_hint:
                        memory_hint_parts.append(hist_hint)
                    decision_hint = build_session_decision_hint(chat_query)
                    if decision_hint:
                        memory_hint_parts.append(decision_hint)
                memory_hint = "\n".join(memory_hint_parts).strip()
                if memory_hint:
                    trace["policy"]["memory_hint"] = memory_hint[:500]

                rag_docs = retrieve_evidence(
                    chat_query,
                    docs=(
                        build_crisis_evidence_docs(data_loader.CRISIS_EVENTS)
                        + build_history_evidence_docs(st.session_state.run_trace_history, max_items=14)
                    ),
                    top_k=4,
                )
                rag_context = format_evidence_block(rag_docs, max_chars=1800)
                st.session_state.rag_last_docs = serialize_retrieved(rag_docs)
                trace["policy"]["rag_doc_count"] = len(rag_docs)
                trace["policy"]["rag_refs"] = [d.reference_id for d in rag_docs]
                if rag_docs:
                    trace_event(trace, "rag_retrieval", f"docs={len(rag_docs)} refs={trace['policy']['rag_refs']}")

                prompt_for_agent = build_structured_prompt(
                    user_query=chat_query,
                    facts_plain=facts_plain,
                    risk_profile=st.session_state.risk_profile,
                    memory_hint=memory_hint,
                    rag_context=rag_context,
                    evidence_gate_strict=bool(st.session_state.evidence_gate_strict),
                )
                if compare_query:
                    prompt_for_agent += (
                        "\n\nAdditional rule for compare mode:\n"
                        "- Treat deterministic compare facts as authoritative.\n"
                        "- Do not run or infer alternative simulations.\n"
                        "- Add only concise commentary and hedging implications from provided facts."
                    )

                def _push_gpt_message(
                    answer_text: str,
                    deployment_used: str,
                    label_strategy: str,
                    cached: bool = False,
                    structured_payload: dict | None = None,
                ):
                    payload = structured_payload or parse_structured_agent_output(answer_text)
                    st.session_state.last_structured_payload = payload
                    if payload:
                        trace["result"]["structured_output_valid"] = True
                        trace["result"]["uncertainty_score"] = payload.get("uncertainty_score")
                        trace["result"]["confidence_reason"] = payload.get("confidence_reason")
                        validation = payload.get("validation", {})
                        if isinstance(validation, dict) and "critic_approved" in validation:
                            trace["result"]["critic_approved"] = bool(validation.get("critic_approved"))
                            if "critic_rounds" in validation:
                                trace["result"]["critic_rounds"] = int(validation.get("critic_rounds") or 0)
                            if isinstance(validation.get("local_evidence_gate"), dict):
                                trace["result"]["local_evidence_gate"] = validation.get("local_evidence_gate")
                        formatted_answer = render_structured_payload_html(payload)
                    else:
                        trace["result"]["structured_output_valid"] = False
                        formatted_answer = format_llm_text_for_card(answer_text)
                        formatted_answer = (
                            "<b>Unstructured output fallback</b><br>"
                            "• Model did not return valid JSON schema.<br><br>"
                            + formatted_answer
                        )
                    body = f"{facts_html}<br><br>{formatted_answer}" if facts_html else formatted_answer
                    suffix = ", cached" if cached else ""
                    st.session_state.agent_messages.append(
                        ("Sentinel", "🛡️", "agent-sentinel",
                         f"<b>Agent ({label_strategy}, {deployment_used}{suffix})</b><br>{body}")
                    )

                def _on_backoff(wait_sec: float, retry_idx: int, max_attempts: int) -> None:
                    _step(
                        70,
                        f"Transient error. Waiting {wait_sec:.1f}s before retry {retry_idx + 1}/{max_attempts}",
                    )
                    trace_event(trace, "gpt_backoff", f"wait={wait_sec:.1f}s retry={retry_idx + 1}/{max_attempts}")

                def _run_candidate(
                    deployment: str,
                    call_strategy: str,
                    call_timeout: int,
                    call_retries: int,
                    eta_fallback: float,
                ) -> tuple[str, dict]:
                    register_gpt_call()
                    answer_text = _run_with_heartbeat(
                        lambda: run_agent_query_with_backoff(
                            prompt_for_agent,
                            timeout_sec=call_timeout,
                            strategy=call_strategy,
                            deployment_name=deployment,
                            max_retries=call_retries,
                            max_total_wait_sec=min(30.0, call_timeout + 10.0),
                            on_backoff=_on_backoff,
                        ),
                        f"Running GPT analysis ({call_strategy})",
                        eta_sec=eta_fallback,
                    )
                    payload = parse_structured_agent_output(answer_text)
                    if not payload:
                        raise RuntimeError("StructuredOutputMissing")
                    return answer_text, payload

                def _critic_gate_or_raise(
                    candidate_answer: str,
                    candidate_payload: dict,
                    deployment: str,
                ) -> tuple[str, dict, dict]:
                    answer_curr = candidate_answer
                    payload_curr = candidate_payload
                    critic_result: dict = {}
                    max_rounds = agentic_ops.critic_round_limit(bool(st.session_state.critic_auto_repair))
                    for gate_round in range(max_rounds):
                        _step(78 + gate_round * 4, "Validating output with Critic gate")
                        local_gate = validate_payload_evidence(
                            payload_curr,
                            allowed_r_refs=set(trace["policy"].get("rag_refs", [])),
                            require_reference_for_numeric_claims=bool(st.session_state.evidence_gate_strict),
                            facts_available=(facts_mode != "none"),
                        )
                        trace["result"]["local_evidence_gate"] = local_gate
                        trace_event(
                            trace,
                            "evidence_gate",
                            f"approved={local_gate['approved']} issues={len(local_gate['issues'])}",
                        )

                        if local_gate["approved"]:
                            critic_result = run_critic_validation(
                                query=chat_query,
                                facts_plain=facts_plain,
                                candidate_json_text=json.dumps(payload_curr),
                                timeout_sec=10,
                                deployment_name=deployment,
                            )
                        else:
                            critic_result = {
                                "approved": False,
                                "issues": local_gate["issues"],
                                "required_fixes": local_gate["required_fixes"],
                                "uncertainty_score": max(
                                    float(payload_curr.get("uncertainty_score", 0.5) or 0.5),
                                    0.7,
                                ),
                                "confidence_reason": "Local evidence gate failed.",
                            }

                        approved = bool(critic_result.get("approved", False))
                        trace_event(trace, "critic_gate", f"approved={approved}")
                        if approved:
                            payload_curr["validation"] = {
                                "critic_approved": True,
                                "critic_rounds": gate_round + 1,
                                "critic_issues": critic_result.get("issues", []),
                                "required_fixes": critic_result.get("required_fixes", []),
                                "local_evidence_gate": local_gate,
                            }
                            if "uncertainty_score" not in payload_curr and critic_result.get("uncertainty_score") is not None:
                                payload_curr["uncertainty_score"] = critic_result.get("uncertainty_score")
                            if "confidence_reason" not in payload_curr and critic_result.get("confidence_reason"):
                                payload_curr["confidence_reason"] = critic_result.get("confidence_reason")
                            return answer_curr, payload_curr, critic_result

                        if gate_round >= max_rounds - 1:
                            break
                        issues = critic_result.get("issues", [])
                        fixes = critic_result.get("required_fixes", [])
                        revision_prompt = (
                            f"{prompt_for_agent}\n\n"
                            "Critic rejected the previous JSON. Revise it now and return strict JSON only.\n"
                            f"Issues: {issues}\n"
                            f"Required fixes: {fixes}\n"
                            f"Previous JSON: {json.dumps(payload_curr)}"
                        )
                        register_gpt_call()
                        answer_curr = _run_with_heartbeat(
                            lambda: run_agent_query_with_backoff(
                                revision_prompt,
                                timeout_sec=max(10, timeout_sec_effective - 4),
                                strategy="commentary_direct" if compare_query else "simple",
                                deployment_name=deployment,
                                max_retries=0,
                                max_total_wait_sec=14.0,
                                on_backoff=_on_backoff,
                            ),
                            "Revising output after critic feedback",
                            eta_sec=8.0,
                        )
                        payload_curr = parse_structured_agent_output(answer_curr)
                        if not payload_curr:
                            raise RuntimeError("CriticGateFailed: revision output not structured.")
                    raise RuntimeError("CriticGateFailed")

                cache_entry, cache_mode = find_cached_agent_response(
                    exact_key=cache_key,
                    query=chat_query,
                    fingerprint=fingerprint,
                )
                if cache_entry and isinstance(cache_entry.get("payload"), dict):
                    cache_gate = validate_payload_evidence(
                        cache_entry["payload"],
                        allowed_r_refs=set(trace["policy"].get("rag_refs", [])),
                        require_reference_for_numeric_claims=bool(st.session_state.evidence_gate_strict),
                        facts_available=(facts_mode != "none"),
                    )
                    if not cache_gate["approved"]:
                        trace_event(
                            trace,
                            "gpt_cache_rejected",
                            f"issues={len(cache_gate['issues'])}",
                        )
                        trace["policy"]["cache_rejection_evidence_gate"] = cache_gate
                        cache_entry = None
                        cache_mode = "none"
                trace["policy"]["cache_hit"] = bool(cache_entry)
                trace["policy"]["cache_mode"] = cache_mode
                if cache_entry:
                    _step(60, "Using cached GPT analysis")
                    _push_gpt_message(
                        answer_text=cache_entry["answer"],
                        deployment_used=cache_entry.get("deployment", primary_deployment),
                        label_strategy=cache_entry.get("strategy", strategy),
                        cached=True,
                        structured_payload=cache_entry.get("payload"),
                    )
                    gpt_success = True
                    register_gpt_success()
                    engine_label = (
                        f"{cache_entry.get('deployment', primary_deployment)} "
                        f"({cache_entry.get('strategy', strategy)}, {cache_mode})"
                    )
                    run_state = "gpt_cached"
                    trace_event(trace, "gpt_cache_hit", engine_label)
                else:
                    try:
                        eta_guess = estimate_eta_seconds(
                            st.session_state.run_trace_history,
                            strategy=strategy,
                            fallback=float(timeout_sec_effective),
                        )
                        _step(60, f"Running GPT analysis ({strategy})")
                        answer, parsed_payload = _run_candidate(
                            deployment=primary_deployment,
                            call_strategy=strategy,
                            call_timeout=timeout_sec_effective,
                            call_retries=retry_policy,
                            eta_fallback=eta_guess,
                        )
                        answer, parsed_payload, critic_result = _critic_gate_or_raise(
                            answer,
                            parsed_payload,
                            deployment=primary_deployment,
                        )
                        _push_gpt_message(answer, primary_deployment, strategy, structured_payload=parsed_payload)
                        st.session_state.agent_response_cache[cache_key] = {
                            "answer": answer,
                            "payload": parsed_payload,
                            "deployment": primary_deployment,
                            "strategy": strategy,
                            "ts": time.time(),
                            "query_tokens": sorted(_tokenize_query(chat_query)),
                            "fingerprint": fingerprint,
                            "critic": critic_result,
                        }
                        gpt_success = True
                        register_gpt_success()
                        engine_label = f"{primary_deployment} ({strategy})"
                        run_state = "gpt_ok"
                        trace_event(trace, "gpt_ok", engine_label)
                    except Exception as exc:
                        trace_event(trace, "gpt_err", f"{type(exc).__name__}: {str(exc)[:160]}")
                        fail_reason = "timeout" if is_timeout_error(exc) else ("rate_limit" if is_rate_limit_error(exc) else "other")
                        if "RunCancelledByUser" in str(exc):
                            fail_reason = "cancelled"

                        if fail_reason == "cancelled":
                            st.session_state.agent_messages.append(
                                ("Sentinel", "🛡️", "agent-sentinel", "Run cancelled by user before GPT completion.")
                            )
                            run_state = "cancelled"
                        else:
                            register_gpt_failure(fail_reason)
                            fallback_ok = False
                            if fail_reason in {"rate_limit", "timeout", "other"}:
                                try:
                                    _step(82, f"Retrying on fallback deployment ({fallback_deployment})")
                                    answer, parsed_payload = _run_candidate(
                                        deployment=fallback_deployment,
                                        call_strategy="commentary_direct" if compare_query else "simple",
                                        call_timeout=max(12, timeout_sec_effective - 4),
                                        call_retries=0,
                                        eta_fallback=10.0,
                                    )
                                    answer, parsed_payload, critic_result = _critic_gate_or_raise(
                                        answer,
                                        parsed_payload,
                                        deployment=fallback_deployment,
                                    )
                                    _push_gpt_message(
                                        answer,
                                        fallback_deployment,
                                        "commentary_direct" if compare_query else "simple",
                                        structured_payload=parsed_payload,
                                    )
                                    st.session_state.agent_response_cache[cache_key] = {
                                        "answer": answer,
                                        "payload": parsed_payload,
                                        "deployment": fallback_deployment,
                                        "strategy": "commentary_direct" if compare_query else "simple",
                                        "ts": time.time(),
                                        "query_tokens": sorted(_tokenize_query(chat_query)),
                                        "fingerprint": fingerprint,
                                        "critic": critic_result,
                                    }
                                    gpt_success = True
                                    register_gpt_success()
                                    engine_label = f"{fallback_deployment} (fallback)"
                                    run_state = "gpt_fallback_ok"
                                    trace_event(trace, "gpt_fallback_ok", engine_label)
                                    fallback_ok = True
                                except Exception as exc_fb:
                                    trace_event(trace, "gpt_fallback_err", f"{type(exc_fb).__name__}: {str(exc_fb)[:140]}")

                            if not fallback_ok:
                                st.session_state.agent_messages.append(
                                    ("Sentinel", "🛡️", "agent-sentinel",
                                     f"⚠️ GPT analysis blocked ({type(exc).__name__}). Using deterministic local output only.")
                                )
                                if parsed and local_sec is None:
                                    _step(88, "GPT unavailable. Running local fallback simulation")
                                    t_local = time.perf_counter()
                                    date = parsed.get("date") or selected_date
                                    G = do_build_network(date, threshold)
                                    do_run_shock(G, parsed["ticker"], parsed["shock"], model_for_query)
                                    local_sec = time.perf_counter() - t_local
                                    trace_event(trace, "local_fallback_done", f"{local_sec:.3f}s")
                                    engine_label = "Local engine (after GPT failure)"
                                    run_state = "gpt_failed_local_fallback"
                                else:
                                    run_state = "gpt_failed"
                        trace["policy"]["cache_hit"] = False
                gpt_sec = time.perf_counter() - t_gpt
                trace_event(trace, "gpt_done", f"{gpt_sec:.3f}s")
            elif st.session_state.agent_mode and parsed and not st.session_state.gpt_for_parseable_queries:
                _step(55, "Fast mode: local result only (GPT skipped)")
                st.session_state.agent_messages.append(
                    ("Sentinel", "🛡️", "agent-sentinel",
                     "⚡ Fast mode active: skipped GPT analysis for standard shock query. "
                     "Enable sidebar option to run GPT here too.")
                )
                run_state = "local_fast_mode"
                trace_event(trace, "local_fast_mode", "gpt skipped for parseable query")
            elif not parsed:
                _step(45, "Unable to parse query locally")
                if gpt_policy_block_reason:
                    st.session_state.agent_messages.append(
                        (
                            "Sentinel",
                            "🛡️",
                            "agent-sentinel",
                            f'GPT unavailable ({html.escape(gpt_policy_block_reason)}), and local parser could not parse: '
                            f'"<i>{chat_query}</i>". Try: "What if JPM crashes 40%?"',
                        )
                    )
                    run_state = "gpt_policy_block_parse_failed"
                    trace_event(trace, "parse_failed_blocked", gpt_policy_block_reason)
                else:
                    st.session_state.agent_messages.append(
                        ("Sentinel", "🛡️", "agent-sentinel",
                         f'Could not parse query: "<i>{chat_query}</i>". '
                         f'Try: "What if JPM crashes 40%?" or "Simulate AAPL 60% shock"')
                    )
                    run_state = "parse_failed"
                    trace_event(trace, "parse_failed", "local parser could not detect ticker")

        elapsed = time.perf_counter() - t_start
        progress.progress(100)
        elapsed_box.caption(f"Elapsed: {elapsed:.1f}s")
        status.update(label=f"Completed in {elapsed:.1f}s", state="complete")

    prev_result = trace.get("result", {}) if isinstance(trace.get("result"), dict) else {}
    st.session_state.last_run_metrics = {
        "total_sec": elapsed,
        "local_sec": local_sec,
        "gpt_sec": gpt_sec,
        "gpt_attempted": gpt_attempted,
        "gpt_success": gpt_success,
        "engine": engine_label,
        "state": run_state,
        "critic_rounds": prev_result.get("critic_rounds"),
        "local_evidence_gate": prev_result.get("local_evidence_gate"),
        "gpt_calls_total_session": st.session_state.gpt_calls_total_session,
        "gpt_rate_limit_hits": st.session_state.gpt_rate_limit_hits,
        "gpt_fail_streak": st.session_state.gpt_fail_streak,
        "gpt_circuit_open_until": st.session_state.gpt_circuit_open_until,
    }
    trace["timings"] = {
        "total_sec": round(elapsed, 3),
        "local_sec": round(local_sec, 3) if isinstance(local_sec, float) else None,
        "gpt_sec": round(gpt_sec, 3) if isinstance(gpt_sec, float) else None,
    }
    trace["result"] = {
        "state": run_state,
        "engine": engine_label,
        "gpt_attempted": gpt_attempted,
        "gpt_success": gpt_success,
        "structured_output_valid": bool(prev_result.get("structured_output_valid", False)),
        "critic_approved": prev_result.get("critic_approved"),
        "critic_rounds": prev_result.get("critic_rounds"),
        "local_evidence_gate": prev_result.get("local_evidence_gate"),
        "uncertainty_score": prev_result.get("uncertainty_score"),
        "confidence_reason": prev_result.get("confidence_reason"),
    }
    trace["policy"]["executor_log"] = summarize_executor_log(trace.get("events", []))
    st.session_state.latest_executor_log = trace["policy"]["executor_log"]
    trace["quality"] = evaluate_run_trace(trace)
    st.session_state.run_quality = trace["quality"]
    advance_workflow(trace, "completed")
    trace_event(trace, "run_complete", run_state)
    trace = finalize_run_trace(trace)
    remember_session_decision(chat_query, trace)
    persist_run_trace(trace)


# Metrics bar
if st.session_state.graph_data:
    gd = st.session_state.graph_data
    m = gd["metrics"]
    sr = st.session_state.shock_result
    gate_info = {}
    if isinstance(st.session_state.run_trace, dict):
        result_now = st.session_state.run_trace.get("result", {})
        if isinstance(result_now, dict) and isinstance(result_now.get("local_evidence_gate"), dict):
            gate_info = result_now.get("local_evidence_gate", {})
    has_gate = bool(gate_info)

    cols_count = 7 if sr else 5
    if has_gate:
        cols_count += 1
    cols = st.columns(cols_count)
    idx = 0
    cols[idx].metric("Nodes", m["n_nodes"])
    idx += 1
    cols[idx].metric("Edges", f"{m['n_edges']:,}")
    idx += 1
    cols[idx].metric("Density", f"{m['density']:.3f}")
    idx += 1
    cols[idx].metric("Regime", gd["regime"])
    idx += 1
    cols[idx].metric("VIX", f"{gd['vix']:.1f}")
    idx += 1
    if has_gate:
        pass_fail = "PASS" if gate_info.get("approved", False) else "FAIL"
        issue_count = len(gate_info.get("issues", []) or [])
        delta = f"{issue_count} issue(s)" if issue_count else "0 issue(s)"
        cols[idx].metric("Evidence Gate", pass_fail, delta=delta)
        idx += 1
    if sr:
        summary = sr.summary()
        cols[idx].metric("Cascade", f"{summary['cascade_depth']} waves")
        idx += 1
        cols[idx].metric("Avg Stress", f"{summary['avg_stress']*100:.1f}%")


with tab_simulate:
    render_stress_lab_tab(_build_stress_lab_tab_context())

with tab_dashboard:
    render_surveillance_tab(_build_surveillance_tab_context())

with tab_outlook:
    render_outlook_tab(_build_outlook_tab_context())

with tab_explain:
    render_audit_trail_tab(_build_audit_trail_tab_context())

with tab_settings:
    st.markdown("### 🤖 Ops and Model Controls")
    is_agent_ready, agent_err = get_agent_config_status()
    access_policy = get_gpt_access_policy()
    can_enable_agent_mode = bool(is_agent_ready and access_policy["allowed"])
    if can_enable_agent_mode:
        st.caption("GPT orchestrator available")
    else:
        st.caption("GPT orchestrator unavailable (using local fallback)")

    st.session_state.agent_mode = st.toggle(
        "Use GPT Orchestrator",
        value=st.session_state.agent_mode if can_enable_agent_mode else False,
        disabled=not can_enable_agent_mode,
        help="When enabled, chat queries are routed to the multi-agent orchestrator (Architect → Quant → Advisor).",
    )
    st.session_state.agent_strategy = st.selectbox(
        "Agent strategy",
        options=["simple", "orchestrator", "workflow_parallel"],
        index=["simple", "orchestrator", "workflow_parallel"].index(st.session_state.agent_strategy)
        if st.session_state.agent_strategy in {"simple", "orchestrator", "workflow_parallel"} else 0,
        disabled=not st.session_state.agent_mode,
        help=(
            "simple = one tool-calling agent. orchestrator = agent-as-tool chain. "
            "workflow_parallel = control-plane workflow "
            "(state machine + deterministic evidence + Planner/Architect+Quant/Critic). "
            "For multi-ticker compare queries, app auto-switches to commentary_direct "
            "(deterministic local compare + GPT commentary only)."
        ),
    )
    st.session_state.high_quality_mode = st.toggle(
        "High quality mode (gpt-4o)",
        value=st.session_state.high_quality_mode,
        disabled=not st.session_state.agent_mode,
        help="OFF = use gpt-4o-mini by default for speed and fewer 429. ON = use gpt-4o.",
    )
    st.session_state.gpt_for_parseable_queries = st.toggle(
        "Run GPT on standard shock queries",
        value=st.session_state.gpt_for_parseable_queries,
        disabled=not st.session_state.agent_mode,
        help="If disabled, standard parseable shock queries use fast local engine only.",
    )
    st.session_state.critic_auto_repair = st.toggle(
        "Critic auto-repair loop",
        value=st.session_state.critic_auto_repair,
        disabled=not st.session_state.agent_mode,
        help="When enabled, failed critic checks trigger one automatic revision pass.",
    )
    st.session_state.evidence_gate_strict = st.toggle(
        "Strict evidence gate",
        value=st.session_state.evidence_gate_strict,
        disabled=not st.session_state.agent_mode,
        help=(
            "When enabled, numeric claims must include E#/R# references and unknown R# refs are rejected."
        ),
    )
    st.session_state.use_session_memory = st.toggle(
        "Use session memory hints",
        value=st.session_state.use_session_memory,
        help="Injects prior decision hints from this session into planner/prompt context.",
    )
    st.session_state.risk_profile = st.selectbox(
        "Risk profile",
        options=["conservative", "balanced", "aggressive"],
        index=["conservative", "balanced", "aggressive"].index(st.session_state.risk_profile)
        if st.session_state.risk_profile in {"conservative", "balanced", "aggressive"} else 1,
        help="Portfolio posture used by Advisor recommendations.",
    )
    st.session_state.agent_timeout_sec = st.slider(
        "Agent timeout (sec)",
        min_value=10,
        max_value=120,
        value=st.session_state.agent_timeout_sec,
        step=5,
        disabled=not st.session_state.agent_mode,
    )

    try:
        primary_dep, fallback_dep = get_deployment_routing(st.session_state.high_quality_mode)
        st.caption(f"Primary: {primary_dep} | Fallback: {fallback_dep}")
    except Exception:
        pass
    if not is_agent_ready and agent_err:
        st.warning(agent_err)

    if st.button("Run GPT Diagnostic", use_container_width=True, disabled=not can_enable_agent_mode):
        try:
            st.session_state.agent_diagnostic = run_gpt_diagnostic()
        except Exception as exc:
            st.session_state.agent_diagnostic = f"diagnostic_err={type(exc).__name__}: {exc}"
    if st.session_state.agent_diagnostic:
        st.code(st.session_state.agent_diagnostic, language="text")

    rate_cfg_caption = (
        f"Soft limits: {_get_runtime_int('GPT_MAX_CALLS_PER_MINUTE_SESSION', 8)}/min session, "
        f"{_get_runtime_int('GPT_MAX_CALLS_PER_MINUTE_GLOBAL', 20)}/min global, "
        f"{_get_runtime_int('GPT_MAX_CALLS_PER_SESSION', 120)} per session, "
        f"{_get_runtime_int('GPT_MAX_CALLS_PER_DAY_GLOBAL', 600)} per day global."
    )
    st.caption(rate_cfg_caption)
    st.caption("Tool contract: mcp.tool.result.v1 (MCP-ready JSON envelopes)")

    st.divider()
    st.markdown("### 🔍 Observability")
    st.session_state.show_explainability = st.toggle(
        "Show explainability panel",
        value=st.session_state.show_explainability,
    )
    st.session_state.persist_trace_logs = st.toggle(
        "Persist traces to JSONL",
        value=st.session_state.persist_trace_logs,
        help="Stores each run trace in artifacts/run_traces.jsonl",
    )
    bcol1, bcol2 = st.columns(2)
    if bcol1.button("🧪 Run Local Benchmark", use_container_width=True):
        with st.spinner("Running benchmark pack..."):
            st.session_state.eval_results = run_local_benchmark(st.session_state.sel_threshold)
    if bcol2.button("🧩 Evaluate Scenario Pack", use_container_width=True):
        st.session_state.scenario_eval_results = run_scenario_pack_eval()


# Footer
st.divider()
st.caption(
    "RiskSentinel — Microsoft AI Dev Days Hackathon 2026 | "
    "Built with Microsoft Agent Framework, Azure AI Foundry, NetworkX, Streamlit"
)
