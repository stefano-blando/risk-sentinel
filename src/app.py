"""
RiskSentinel — Streamlit App
Agentic Systemic Risk Simulator for Financial Contagion
"""

import asyncio
import concurrent.futures
import sys
import html
import time
import json
import os
import hmac
import hashlib
import io
import zipfile
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

from src.agents.orchestrator import (
    create_orchestrator,
    create_simple_agent,
    run_query,
    run_parallel_workflow,
)
from src.agents.critic import create_critic_agent
from src.agents.evaluation import EvalSample, evaluate_samples
from src.agents.evidence_rag import (
    build_crisis_evidence_docs,
    build_history_evidence_docs,
    format_evidence_block,
    retrieve_evidence,
    serialize_retrieved,
)
from src.agents.evidence_validation import validate_payload_evidence
from src.core import data_loader, network, contagion
from src import agentic_ops, reporting, ui_panels
from src.utils.azure_config import (
    get_agent_framework_chat_client,
    get_openai_client,
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
        gap: 0.35rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: var(--surface-0);
        border: 1px solid var(--border-subtle);
        border-bottom: none;
        border-radius: 10px 10px 0 0;
        color: var(--text-muted);
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
defaults = {
    "graph_data": None, "shock_result": None, "agent_messages": [],
    "pos": None, "current_wave": -1,
    "sector_dict": None, "tickers": None,
    "chat_history": [], "comparison": None,
    "sel_date": None, "sel_ticker": "JPM", "sel_shock": 50,
    "sel_model": "debtrank", "sel_threshold": 0.5,
    "agent_mode": True, "agent_timeout_sec": 35, "agent_strategy": "simple",
    "high_quality_mode": False,
    "gpt_for_parseable_queries": False,
    "agent_diagnostic": "",
    "last_run_metrics": None,
    "agent_response_cache": {},
    "demo_mode": False,
    "demo_story": list(DEMO_QUERIES.keys())[0],
    "pending_chat_query": "",
    "run_trace": None,
    "run_trace_history": [],
    "persist_trace_logs": True,
    "show_explainability": True,
    "eval_results": None,
    "judge_unlocked": False,
    "judge_unlock_error": "",
    "gpt_rate_events": [],
    "gpt_calls_total_session": 0,
    "gpt_rate_limit_hits": 0,
    "risk_profile": "balanced",
    "scenario_pack_choice": SCENARIO_PACK[0]["name"],
    "last_structured_payload": None,
    "run_quality": None,
    "scenario_eval_results": None,
    "compare_rows_local": [],
    "compare_meta": {},
    "gpt_fail_streak": 0,
    "gpt_circuit_open_until": 0.0,
    "run_cancel_requested": False,
    "rag_last_docs": [],
    "judge_kpis": {},
    "critic_auto_repair": True,
    "evidence_gate_strict": True,
    "use_session_memory": True,
    "session_decisions": [],
    "commander_results": None,
    "autonomous_results": None,
    "portfolio_text": "",
    "auto_portfolio_n": 5,
    "portfolio_copilot": None,
    "latest_policy_plan": [],
    "latest_executor_log": [],
    "last_agentic_action": "",
    "agentic_ops_cache": {},
    "full_demo_last_run": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

AUTO_BUILD_ON_START = os.getenv("AUTO_BUILD_ON_START", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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
            f"- RISKSENTINEL_DATA_ROOT: `{paths['env_data_root'] or '(not set)'}`\n\n"
            "Set `RISKSENTINEL_DATA_ROOT` to a folder containing the processed dataset "
            "(including `sector_mapping.parquet` and `networks/node_centralities.pkl`).\n\n"
            f"Technical error: `{type(exc).__name__}: {exc}`"
        )
        st.stop()


# ---------------------------------------------------------------------------
# NATURAL LANGUAGE PARSER (local, no LLM needed)
# ---------------------------------------------------------------------------
def extract_tickers_from_query(query: str, tickers: list[str]) -> list[str]:
    """Extract one or more tickers from user query using aliases + direct symbols."""
    query_upper = query.upper().strip()
    found: list[str] = []

    for name, ticker in COMPANY_NAME_MAP.items():
        if name in query_upper and ticker not in found:
            found.append(ticker)

    for ticker in tickers:
        if re.search(rf"\b{re.escape(ticker)}\b", query_upper) and ticker not in found:
            found.append(ticker)

    return found


def parse_chat_query(query: str) -> dict | None:
    """Parse natural language shock queries into parameters.
    Returns dict with ticker, shock, date or None if not parseable.
    """
    tickers = st.session_state.tickers

    found_tickers = extract_tickers_from_query(query, tickers)
    if not found_tickers:
        return None

    # Find shock percentage
    shock = 50  # default
    pct_match = re.search(r'(\d+)\s*%', query)
    if pct_match:
        shock = min(100, max(10, int(pct_match.group(1))))

    # Find date
    date = None
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
    if date_match:
        date = date_match.group(1)

    return {"ticker": found_tickers[0], "tickers": found_tickers, "shock": shock, "date": date}


def infer_model_from_query(query: str) -> str:
    q = query.lower()
    if "linear threshold" in q or "linear_threshold" in q:
        return "linear_threshold"
    if "cascade removal" in q or "cascade_removal" in q:
        return "cascade_removal"
    return "debtrank"


def normalize_chat_query(query: str) -> str:
    """Normalize user query text for cleaner UI/logging."""
    q = " ".join(query.strip().split())
    if len(q) >= 2 and ((q[0] == q[-1] == '"') or (q[0] == q[-1] == "'")):
        q = q[1:-1].strip()
    return q


def is_complex_query(query: str) -> bool:
    """Heuristic: detect prompts that need comparative/strategic GPT reasoning."""
    q = query.lower()
    signals = [
        "compare",
        "comparison",
        " vs ",
        "versus",
        "difference",
        "differences",
        "portfolio",
        "hedging plan",
        "strategy",
        "overweight",
        "underweight",
        "concentration",
    ]
    return any(sig in q for sig in signals)


def is_compare_query(query: str, parsed: dict | None) -> bool:
    if not parsed or len(parsed.get("tickers", [])) < 2:
        return False
    q = query.lower()
    return any(sig in q for sig in ["compare", "rank", "difference", "versus", " vs ", "between"])


def is_query_in_scope(query: str, parsed: dict | None) -> tuple[bool, str]:
    """Guardrail: keep assistant focused on network/crisis/contagion scope."""
    if parsed:
        return True, "Parsed ticker/shock scenario."

    q = query.lower()
    scope_signals = [
        "network",
        "regime",
        "crisis",
        "contagion",
        "cascade",
        "debtrank",
        "shock",
        "crash",
        "systemic risk",
        "financial risk",
        "hedg",
        "sector",
        "vix",
        "stock",
        "ticker",
    ]
    if any(sig in q for sig in scope_signals):
        return True, "Detected network/crisis/contagion intent."
    return False, "Out of scope for RiskSentinel domain."


def _get_runtime_value(name: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then env vars."""
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return str(os.getenv(name, default)).strip()


def _get_runtime_int(name: str, default: int) -> int:
    raw = _get_runtime_value(name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


@st.cache_resource
def get_global_gpt_rate_bucket() -> dict:
    """Shared in-process bucket for soft global GPT throttle."""
    return {"events": []}


def _prune_events(events: list[float], now_ts: float, window_sec: int = 60) -> list[float]:
    return [ts for ts in events if (now_ts - ts) <= window_sec]


def _agentic_cache_key(op_name: str, **kwargs) -> str:
    raw = json.dumps({"op": op_name, **kwargs}, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _run_agentic_operation(
    *,
    op_name: str,
    cache_key: str,
    fn: Callable[[], dict],
    timeout_sec: int = AGENTIC_OP_TIMEOUT_SEC,
    ttl_sec: int = AGENTIC_CACHE_TTL_SEC,
) -> tuple[dict, bool]:
    """Run deterministic agentic op with timeout + session cache."""
    now = time.time()
    cache = st.session_state.agentic_ops_cache
    row = cache.get(cache_key)
    if isinstance(row, dict) and (now - float(row.get("ts", 0.0))) <= ttl_sec:
        return dict(row.get("data", {})), True

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            result = future.result(timeout=max(3, int(timeout_sec)))
        except concurrent.futures.TimeoutError:
            future.cancel()
            return {
                "ok": False,
                "error": f"{op_name} timed out after {timeout_sec}s",
                "timeout": True,
            }, False
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }, False

    if not isinstance(result, dict):
        result = {"ok": False, "error": f"{op_name} returned non-dict payload"}
    if "ok" not in result:
        result["ok"] = True
    cache[cache_key] = {"ts": now, "data": result}
    st.session_state.agentic_ops_cache = cache
    return result, False


def _tokenize_query(query: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[a-zA-Z]{2,}", query.lower())
        if tok not in QUERY_TOKEN_STOPWORDS
    }


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def build_cache_fingerprint(
    *,
    parsed: dict | None,
    threshold: float,
    model: str,
    risk_profile: str,
    schema_version: str,
    strategy: str,
) -> dict:
    tickers = sorted((parsed or {}).get("tickers", []))
    return {
        "route": "compare" if len(tickers) >= 2 else "single",
        "tickers": tickers,
        "ticker": (parsed or {}).get("ticker"),
        "shock": (parsed or {}).get("shock"),
        "date": (parsed or {}).get("date"),
        "threshold": round(float(threshold), 3),
        "model": model,
        "risk_profile": risk_profile,
        "schema_version": schema_version,
        "strategy": strategy,
    }


def get_gpt_access_policy() -> dict:
    """Judge access policy: open by default, locked only if code is configured."""
    judge_code = _get_runtime_value("JUDGE_ACCESS_CODE", "")
    gate_enabled = bool(judge_code)
    unlocked = bool(st.session_state.judge_unlocked)
    allowed = (not gate_enabled) or unlocked
    reason = "open" if not gate_enabled else ("unlocked" if unlocked else "judge_code_required")
    return {"gate_enabled": gate_enabled, "allowed": allowed, "reason": reason}


def unlock_judge_access(user_code: str) -> bool:
    expected = _get_runtime_value("JUDGE_ACCESS_CODE", "")
    if not expected:
        st.session_state.judge_unlocked = True
        return True
    ok = hmac.compare_digest(user_code.strip(), expected.strip())
    st.session_state.judge_unlocked = bool(ok)
    return ok


def check_gpt_rate_limit() -> tuple[bool, str]:
    """Soft limiter to keep demo stable and avoid quota spikes."""
    max_session = _get_runtime_int("GPT_MAX_CALLS_PER_SESSION", 120)
    max_per_min_session = _get_runtime_int("GPT_MAX_CALLS_PER_MINUTE_SESSION", 8)
    max_per_min_global = _get_runtime_int("GPT_MAX_CALLS_PER_MINUTE_GLOBAL", 20)

    now = time.time()
    st.session_state.gpt_rate_events = _prune_events(st.session_state.gpt_rate_events, now, 60)
    bucket = get_global_gpt_rate_bucket()
    bucket["events"] = _prune_events(bucket.get("events", []), now, 60)

    if st.session_state.gpt_calls_total_session >= max_session:
        return False, f"session_cap_reached:{max_session}"
    if len(st.session_state.gpt_rate_events) >= max_per_min_session:
        return False, f"session_rate_limit:{max_per_min_session}/min"
    if len(bucket["events"]) >= max_per_min_global:
        return False, f"global_rate_limit:{max_per_min_global}/min"
    return True, "ok"


def register_gpt_call() -> None:
    now = time.time()
    st.session_state.gpt_rate_events = _prune_events(st.session_state.gpt_rate_events, now, 60)
    st.session_state.gpt_rate_events.append(now)
    st.session_state.gpt_calls_total_session += 1
    bucket = get_global_gpt_rate_bucket()
    bucket["events"] = _prune_events(bucket.get("events", []), now, 60)
    bucket["events"].append(now)


def is_gpt_circuit_open() -> tuple[bool, float]:
    now = time.time()
    open_until = float(st.session_state.get("gpt_circuit_open_until", 0.0) or 0.0)
    if open_until > now:
        return True, open_until - now
    return False, 0.0


def register_gpt_success() -> None:
    st.session_state.gpt_fail_streak = 0
    st.session_state.gpt_circuit_open_until = 0.0


def register_gpt_failure(reason: str) -> None:
    streak = int(st.session_state.get("gpt_fail_streak", 0)) + 1
    st.session_state.gpt_fail_streak = streak
    if streak >= 3 or reason in {"rate_limit", "timeout"}:
        st.session_state.gpt_circuit_open_until = max(
            float(st.session_state.get("gpt_circuit_open_until", 0.0) or 0.0),
            time.time() + CIRCUIT_COOLDOWN_SEC,
        )


def estimate_eta_seconds(history: list[dict], strategy: str, fallback: float = 18.0) -> float:
    vals = []
    for row in history[-20:]:
        if row.get("policy", {}).get("router", {}).get("effective_strategy") != strategy:
            continue
        gpt_sec = row.get("timings", {}).get("gpt_sec")
        if isinstance(gpt_sec, (int, float)) and gpt_sec > 0:
            vals.append(float(gpt_sec))
    if not vals:
        return fallback
    return float(np.percentile(vals, 75))


def create_run_trace(
    query: str,
    parsed: dict | None,
    complex_query: bool,
    in_scope: bool,
    scope_reason: str,
    model_for_query: str,
    threshold: float,
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": f"run-{int(now.timestamp() * 1000)}",
        "created_at_utc": now.isoformat(),
        "query": query,
        "parsed": parsed or {},
        "complex_query": complex_query,
        "in_scope": in_scope,
        "scope_reason": scope_reason,
        "model": model_for_query,
        "threshold": threshold,
        "policy": {},
        "timings": {},
        "events": [],
        "result": {},
        "workflow": {"state": "received", "history": ["received"]},
    }


def trace_event(trace: dict, label: str, detail: str = "") -> None:
    trace["events"].append(
        {
            "t_sec": round(time.perf_counter() - trace["_t0"], 3),
            "label": label,
            "detail": detail,
        }
    )


def advance_workflow(trace: dict, next_state: str) -> None:
    wf = trace.setdefault("workflow", {"state": "received", "history": ["received"]})
    cur = wf.get("state", "received")
    allowed = WORKFLOW_TRANSITIONS.get(cur, set())
    if next_state in allowed or cur == next_state:
        wf["state"] = next_state
        wf.setdefault("history", []).append(next_state)
        trace_event(trace, "workflow_state", f"{cur}->{next_state}")
    else:
        wf.setdefault("history", []).append(f"invalid:{cur}->{next_state}")
        trace_event(trace, "workflow_invalid_transition", f"{cur}->{next_state}")


def finalize_run_trace(trace: dict) -> dict:
    trace.pop("_t0", None)
    trace["events"] = trace.get("events", [])[-30:]
    st.session_state.run_trace = trace
    history = st.session_state.run_trace_history
    history.append(trace)
    st.session_state.run_trace_history = history[-50:]
    return trace


def persist_run_trace(trace: dict) -> None:
    """Append trace to local JSONL for post-demo analysis."""
    if not st.session_state.persist_trace_logs:
        return
    try:
        log_dir = Path(__file__).resolve().parents[1] / "artifacts"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "run_traces.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=True) + "\n")
    except Exception:
        # Logging must never break the app flow.
        pass


def run_local_benchmark(threshold: float) -> dict:
    """Quick benchmark pack for hackathon rehearsals (local deterministic)."""
    preserved = {
        "graph_data": st.session_state.graph_data,
        "pos": st.session_state.pos,
        "shock_result": st.session_state.shock_result,
        "current_wave": st.session_state.current_wave,
        "agent_messages": list(st.session_state.agent_messages),
        "comparison": st.session_state.comparison,
    }
    rows = []
    t0 = time.perf_counter()
    ok = 0
    try:
        for q in BENCHMARK_QUERIES:
            q_start = time.perf_counter()
            parsed = parse_chat_query(q)
            if not parsed:
                rows.append(
                    {
                        "query": q,
                        "status": "parse_failed",
                        "latency_s": round(time.perf_counter() - q_start, 3),
                    }
                )
                continue

            model = infer_model_from_query(q)
            date = parsed.get("date") or "2025-12-01"
            try:
                G = do_build_network(date, threshold, emit_messages=False)
                result = contagion.run_shock_scenario(G, parsed["ticker"], parsed["shock"] / 100.0, model)
                s = result.summary()
                ok += 1
                rows.append(
                    {
                        "query": q,
                        "status": "ok",
                        "ticker": parsed["ticker"],
                        "shock": parsed["shock"],
                        "model": model,
                        "waves": s["cascade_depth"],
                        "affected": s["n_affected"],
                        "avg_stress_pct": round(s["avg_stress"] * 100, 2),
                        "latency_s": round(time.perf_counter() - q_start, 3),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "query": q,
                        "status": f"err:{type(exc).__name__}",
                        "latency_s": round(time.perf_counter() - q_start, 3),
                    }
                )
    finally:
        st.session_state.graph_data = preserved["graph_data"]
        st.session_state.pos = preserved["pos"]
        st.session_state.shock_result = preserved["shock_result"]
        st.session_state.current_wave = preserved["current_wave"]
        st.session_state.agent_messages = preserved["agent_messages"]
        st.session_state.comparison = preserved["comparison"]

    total = time.perf_counter() - t0
    latencies = [r["latency_s"] for r in rows]
    return {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(BENCHMARK_QUERIES),
        "n_ok": ok,
        "success_rate_pct": round((ok / max(1, len(BENCHMARK_QUERIES))) * 100, 1),
        "avg_latency_s": round(float(np.mean(latencies)) if latencies else 0.0, 3),
        "total_time_s": round(total, 3),
        "rows": rows,
    }


def run_scenario_pack_eval() -> dict:
    """Evaluate routing expectations for curated judge scenario pack."""
    rows = []
    for scenario in SCENARIO_PACK:
        q = scenario["query"]
        parsed = parse_chat_query(q)
        in_scope, _ = is_query_in_scope(q, parsed)
        complex_query = is_complex_query(q)
        policy = choose_execution_policy(
            parsed=parsed,
            complex_query=complex_query,
            in_scope=in_scope,
            agent_mode=st.session_state.agent_mode,
            gpt_for_parseable_queries=st.session_state.gpt_for_parseable_queries,
            access_allowed=get_gpt_access_policy()["allowed"],
            selected_strategy=st.session_state.agent_strategy,
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
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    return {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": len(rows),
        "n_pass": pass_count,
        "pass_rate_pct": round(pass_count / max(1, len(rows)) * 100, 1),
        "rows": rows,
    }


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


def extract_json_payload(text: str) -> dict | None:
    """Best-effort JSON extraction from model output (raw or fenced)."""
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    obj = re.search(r"(\{[\s\S]*\})", text)
    if obj:
        candidates.append(obj.group(1).strip())

    for raw in candidates:
        try:
            val = json.loads(raw)
            if isinstance(val, dict):
                return val
        except Exception:
            continue
    return None


def parse_structured_agent_output(text: str) -> dict | None:
    payload = extract_json_payload(text)
    if not payload:
        return None

    required = ["situation", "quant_results", "risk_rating", "actions", "monitoring_triggers"]
    if not all(k in payload for k in required):
        return None

    def _as_list(val) -> list[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    parsed = {
        "schema_version": str(payload.get("schema_version", STRUCTURED_SCHEMA_VERSION)),
        "situation": _as_list(payload.get("situation")),
        "quant_results": _as_list(payload.get("quant_results")),
        "risk_rating": str(payload.get("risk_rating", "UNKNOWN")).upper(),
        "actions": _as_list(payload.get("actions")),
        "monitoring_triggers": _as_list(payload.get("monitoring_triggers")),
        "evidence_used": _as_list(payload.get("evidence_used")),
        "notes": str(payload.get("notes", "")).strip(),
        "insufficient_data": bool(payload.get("insufficient_data", False)),
        "uncertainty_score": float(payload.get("uncertainty_score", 0.5))
        if str(payload.get("uncertainty_score", "")).strip() != "" else 0.5,
        "confidence_reason": str(payload.get("confidence_reason", "")).strip(),
        "validation": payload.get("validation", {}) if isinstance(payload.get("validation"), dict) else {},
    }
    parsed["uncertainty_score"] = max(0.0, min(1.0, parsed["uncertainty_score"]))
    return parsed


def render_structured_payload_html(payload: dict) -> str:
    """Render structured JSON output in clean card-friendly HTML."""
    sections = []

    def _section(title: str, items: list[str]) -> None:
        if not items:
            return
        body = "<br>".join(f"• {html.escape(item)}" for item in items)
        sections.append(f"<b>{html.escape(title)}</b><br>{body}")

    _section("Situation", payload.get("situation", []))
    _section("Quant Results", payload.get("quant_results", []))
    sections.append(f"<b>Risk Rating</b><br>• {html.escape(payload.get('risk_rating', 'UNKNOWN'))}")
    _section("Actions", payload.get("actions", []))
    _section("Monitoring Triggers", payload.get("monitoring_triggers", []))
    _section("Evidence Used", payload.get("evidence_used", []))
    uncertainty = payload.get("uncertainty_score")
    if isinstance(uncertainty, (int, float)):
        sections.append(f"<b>Uncertainty</b><br>• {float(uncertainty):.2f}")
    confidence_reason = payload.get("confidence_reason", "")
    if confidence_reason:
        sections.append(f"<b>Confidence Reason</b><br>• {html.escape(str(confidence_reason))}")

    notes = payload.get("notes", "")
    if notes:
        sections.append(f"<b>Notes</b><br>• {html.escape(notes)}")
    return "<br><br>".join(sections)


def build_context_facts_html() -> str:
    """Fallback deterministic facts when no parsed shock scenario is present."""
    gd = st.session_state.graph_data
    if not gd:
        return ""
    m = gd.get("metrics", {})
    lines = [
        "<b>Context Facts (Deterministic)</b>",
        (
            f"• Date: {html.escape(str(gd.get('date', 'n/a')))} | "
            f"Regime: {html.escape(str(gd.get('regime', 'n/a')))} (VIX {gd.get('vix', 0.0):.1f})"
        ),
        (
            f"• Network: nodes {m.get('n_nodes', 'n/a')} | edges {m.get('n_edges', 'n/a')} | "
            f"density {m.get('density', 0.0):.3f}"
        ),
    ]
    return "<br>".join(lines)


def build_memory_hint(query: str, history: list[dict], top_k: int = 2) -> str:
    """Retrieve brief episodic memory from prior runs using token overlap."""
    tokens = {w for w in re.findall(r"[a-zA-Z]{3,}", query.lower()) if w not in {"what", "with", "that", "from"}}
    if not tokens:
        return ""
    scored: list[tuple[int, dict]] = []
    for row in history:
        q = str(row.get("query", "")).lower()
        if not q:
            continue
        q_tokens = set(re.findall(r"[a-zA-Z]{3,}", q))
        score = len(tokens & q_tokens)
        if score <= 0:
            continue
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for _, row in scored[:top_k]:
        result = row.get("result", {})
        timings = row.get("timings", {})
        lines.append(
            f"- prior_query='{row.get('query', '')[:80]}' | state={result.get('state', 'n/a')} | "
            f"gpt_success={result.get('gpt_success', False)} | total_sec={timings.get('total_sec', 'n/a')}"
        )
    return "\n".join(lines)


def build_structured_prompt(
    user_query: str,
    facts_plain: str,
    risk_profile: str,
    memory_hint: str = "",
    rag_context: str = "",
    evidence_gate_strict: bool = True,
) -> str:
    profile_hint = RISK_PROFILE_GUIDANCE.get(risk_profile, RISK_PROFILE_GUIDANCE["balanced"])
    memory_block = f"Episodic memory hints:\n{memory_hint}\n\n" if memory_hint else ""
    rag_block = f"Retrieved evidence context (RAG):\n{rag_context}\n\n" if rag_context else ""
    schema = (
        '{'
        '"schema_version":"v1",'
        '"situation":["..."],'
        '"quant_results":["..."],'
        '"risk_rating":"LOW|ELEVATED|HIGH|CRITICAL",'
        '"actions":["..."],'
        '"monitoring_triggers":["..."],'
        '"evidence_used":["..."],'
        '"notes":"...",'
        '"insufficient_data":false,'
        '"uncertainty_score":0.2,'
        '"confidence_reason":"..."'
        '}'
    )
    parts = [
        "Return ONLY valid JSON (no markdown, no prose outside JSON).\n",
        f"JSON schema example:\n{schema}\n\n",
        "Rules:\n",
        "- Use only values from deterministic facts.\n",
        "- Do not invent numbers.\n",
        "- Keep each list concise (max 4 items).\n",
        "- Return uncertainty_score between 0.0 and 1.0.\n",
        f"- Risk profile to optimize for: {risk_profile} ({profile_hint})\n",
        "- In evidence_used, include only E#/R# references actually used.\n",
        (
            "- Any numeric claim must be traceable via evidence_used references.\n"
            if evidence_gate_strict
            else "- Numeric claim references are recommended where available.\n"
        ),
        "- If RAG context is used, cite only the retrieved R# references.\n",
        "- If facts are insufficient, set insufficient_data=true and explain in notes.\n\n",
        f"Deterministic facts:\n{facts_plain}\n\n",
    ]
    if memory_block:
        parts.append(memory_block)
    if rag_block:
        parts.append(rag_block)
    parts.append(f"User request:\n{user_query}")
    return "".join(parts)


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
    """Store concise session memory record for next runs."""
    result = trace.get("result", {}) if isinstance(trace, dict) else {}
    policy = trace.get("policy", {}) if isinstance(trace, dict) else {}
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": query[:220],
        "state": result.get("state", "n/a"),
        "risk_profile": policy.get("risk_profile", st.session_state.risk_profile),
        "route": (policy.get("router") or {}).get("route", "n/a") if isinstance(policy.get("router"), dict) else "n/a",
        "critic_approved": result.get("critic_approved"),
        "uncertainty_score": result.get("uncertainty_score"),
    }
    mem = list(st.session_state.session_decisions or [])
    mem.append(entry)
    st.session_state.session_decisions = mem[-40:]


def build_session_decision_hint(query: str, top_k: int = 2) -> str:
    """Semantic hint from recent decisions in this Streamlit session."""
    records = st.session_state.session_decisions or []
    if not records:
        return ""
    q_tokens = _tokenize_query(query)
    scored: list[tuple[float, dict]] = []
    for row in records:
        prev_q = str(row.get("query", ""))
        prev_tokens = _tokenize_query(prev_q)
        score = _jaccard_similarity(q_tokens, prev_tokens)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for _, row in scored[:top_k]:
        lines.append(
            f"- memory_query='{row.get('query', '')[:80]}' | state={row.get('state')} | "
            f"risk_profile={row.get('risk_profile')} | critic_approved={row.get('critic_approved')}"
        )
    return "\n".join(lines)


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
    """Compute per-run quality/evaluation metrics."""
    policy = trace.get("policy", {})
    result = trace.get("result", {})
    timings = trace.get("timings", {})
    events = trace.get("events", [])
    state = result.get("state", "")
    gpt_attempted = bool(result.get("gpt_attempted"))
    gpt_success = bool(result.get("gpt_success"))
    structured_ok = bool(result.get("structured_output_valid", False))
    critic_approved = result.get("critic_approved")
    if critic_approved is not None:
        critic_approved = bool(critic_approved)
    facts_mode = policy.get("facts_mode", "none")
    factual_consistency = None
    if gpt_attempted and gpt_success:
        factual_consistency = bool(structured_ok and facts_mode != "none")

    rate_limit_events = sum(1 for e in events if e.get("label") in {"gpt_backoff", "gpt_policy_block"})
    has_local_output = bool(trace.get("parsed")) or bool(policy.get("router", {}).get("run_local_first", False))
    used_fallback = state in {
        "gpt_retry_ok",
        "gpt_fallback_ok",
        "gpt_failed_local_fallback",
        "gpt_policy_block_local",
    } or (state == "gpt_failed" and has_local_output)
    model_uncertainty = result.get("uncertainty_score")
    if isinstance(model_uncertainty, (int, float)):
        uncertainty_score = max(0.0, min(1.0, float(model_uncertainty)))
    else:
        uncertainty_score = 0.15
        if not structured_ok:
            uncertainty_score += 0.25
        if used_fallback:
            uncertainty_score += 0.25
        if rate_limit_events > 0:
            uncertainty_score += 0.15
        if policy.get("facts_mode", "none") == "none":
            uncertainty_score += 0.15
        uncertainty_score = min(1.0, uncertainty_score)
    return {
        "latency_sec": float(timings.get("total_sec", 0.0) or 0.0),
        "gpt_attempted": gpt_attempted,
        "gpt_success": gpt_success,
        "critic_approved": critic_approved,
        "structured_output_valid": structured_ok,
        "factual_consistency": factual_consistency,
        "cache_hit": bool(policy.get("cache_hit", False)),
        "rate_limit_events": rate_limit_events,
        "used_fallback": used_fallback,
        "uncertainty_score": round(float(uncertainty_score), 3),
        "confidence_score": round(float(1.0 - uncertainty_score), 3),
    }


def summarize_quality(history: list[dict]) -> dict:
    if not history:
        return {}
    eval_rows = [h.get("quality", {}) for h in history if h.get("quality")]
    if not eval_rows:
        return {}
    n = len(eval_rows)
    factual_rows = [r for r in eval_rows if r.get("factual_consistency") is not None]
    factual_ok = sum(1 for r in factual_rows if r.get("factual_consistency"))
    return {
        "runs": n,
        "avg_latency_sec": round(float(np.mean([r.get("latency_sec", 0.0) for r in eval_rows])), 2),
        "cache_hit_rate_pct": round(sum(1 for r in eval_rows if r.get("cache_hit")) / n * 100, 1),
        "fallback_rate_pct": round(sum(1 for r in eval_rows if r.get("used_fallback")) / n * 100, 1),
        "gpt_success_rate_pct": round(sum(1 for r in eval_rows if r.get("gpt_success")) / n * 100, 1),
        "rate_limit_events_total": int(sum(int(r.get("rate_limit_events", 0)) for r in eval_rows)),
        "factual_consistency_pct": round((factual_ok / max(1, len(factual_rows))) * 100, 1) if factual_rows else None,
        "avg_uncertainty": round(float(np.mean([r.get("uncertainty_score", 0.5) for r in eval_rows])), 3),
    }


def build_judge_kpis(history: list[dict]) -> dict:
    """Compute judge-facing KPI snapshot from run history."""
    samples: list[EvalSample] = []
    for row in history:
        quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        timings = row.get("timings", {}) if isinstance(row.get("timings"), dict) else {}

        critic_approved = result.get("critic_approved")
        if not isinstance(critic_approved, bool):
            critic_approved = bool(
                result.get("structured_output_valid", False) and result.get("gpt_success", False)
            )

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


def build_submission_bundle_bytes() -> bytes:
    report = generate_report_text()
    brief_md = generate_report_markdown()
    action_ceo = generate_action_pack_ceo_brief()
    action_runbook = generate_action_pack_runbook()
    action_json = generate_action_pack_machine_json()
    trace_json = generate_trace_bundle_json()
    quality = summarize_quality(st.session_state.run_trace_history)
    judge_kpis = build_judge_kpis(st.session_state.run_trace_history)
    quality_json = json.dumps(quality, indent=2)
    judge_json = json.dumps(judge_kpis, indent=2)
    scenario_eval_json = json.dumps(st.session_state.scenario_eval_results or {}, indent=2)
    rag_json = json.dumps(st.session_state.rag_last_docs or [], indent=2)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.txt", report)
        zf.writestr("executive_brief.md", brief_md)
        zf.writestr("explainability_trace.json", trace_json)
        zf.writestr("kpi_snapshot.json", quality_json)
        zf.writestr("judge_dashboard_kpis.json", judge_json)
        zf.writestr("evidence_rag_last_docs.json", rag_json)
        zf.writestr("scenario_pack_eval.json", scenario_eval_json)
        zf.writestr("action_pack_ceo_brief.md", action_ceo)
        zf.writestr("action_pack_risk_runbook.md", action_runbook)
        zf.writestr("action_pack_machine.json", action_json)
    return buf.getvalue()


def format_llm_text_for_card(text: str) -> str:
    """Render LLM markdown-like output as readable HTML for message cards."""
    out_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out_lines.append("")
            continue

        # Remove markdown emphasis/code markers that look noisy in cards.
        line = line.replace("**", "").replace("`", "")

        # Headings
        if line.startswith("### "):
            out_lines.append(f"<b>{html.escape(line[4:])}</b>")
            continue
        if line.startswith("## "):
            out_lines.append(f"<b>{html.escape(line[3:])}</b>")
            continue
        if line.startswith("# "):
            out_lines.append(f"<b>{html.escape(line[2:])}</b>")
            continue

        # Simple markdown tables -> compact text rows
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and all(set(c) <= set("-:") for c in cells):
                continue
            row = " - ".join(c for c in cells if c)
            out_lines.append(html.escape(row))
            continue

        # Bullets
        if line.startswith("- "):
            out_lines.append(f"• {html.escape(line[2:])}")
            continue
        if line.startswith("* "):
            out_lines.append(f"• {html.escape(line[2:])}")
            continue

        out_lines.append(html.escape(line))

    return "<br>".join(out_lines)


def build_simulation_facts_html() -> str:
    """Build deterministic simulation facts block from local engine state."""
    gd = st.session_state.graph_data
    sr = st.session_state.shock_result
    if not gd or not sr:
        return ""

    s = sr.summary()
    threshold = gd.get("threshold")
    threshold_txt = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "n/a"
    facts = [
        "<b>Simulation Facts (Deterministic)</b>",
        (
            f"• Date: {html.escape(str(gd.get('date', 'n/a')))} | "
            f"Threshold: {threshold_txt} | Regime: {html.escape(str(gd.get('regime', 'n/a')))} "
            f"(VIX {gd.get('vix', 0.0):.1f})"
        ),
        (
            f"• Scenario: {html.escape(s['shocked_node'])} shock "
            f"{s['shock_magnitude'] * 100:.0f}% with {html.escape(s['model'])}"
        ),
        (
            f"• Cascade: {s['cascade_depth']} waves | Affected: {s['n_affected']} | "
            f"Defaulted: {s['n_defaulted']}"
        ),
        f"• Total stress: {s['total_stress']:.2f} | Avg stress: {s['avg_stress'] * 100:.2f}%",
    ]
    return "<br>".join(facts)


def _compute_compare_rows(
    G: nx.Graph,
    tickers: list[str],
    shock_pct: int,
    model: str,
) -> tuple[list[dict], dict[str, contagion.ShockResult]]:
    """Compute deterministic metrics for multi-ticker comparison on same graph."""
    sector_dict = st.session_state.sector_dict
    rows: list[dict] = []
    result_by_ticker: dict[str, contagion.ShockResult] = {}
    for ticker in tickers[:MAX_COMPARE_TICKERS]:
        if ticker not in G:
            continue
        result = contagion.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
        result_by_ticker[ticker] = result
        s = result.summary()

        sector_stress: dict[str, float] = {}
        for node, stress in result.node_stress.items():
            if stress <= 0:
                continue
            sector = sector_dict.get(node, "Unknown")
            sector_stress[sector] = sector_stress.get(sector, 0.0) + stress
        top_sectors = sorted(sector_stress.items(), key=lambda x: x[1], reverse=True)[:2]
        top_sector_text = ", ".join(f"{sec} {val:.2f}" for sec, val in top_sectors) or "n/a"

        rows.append(
            {
                "ticker": ticker,
                "cascade_depth": s["cascade_depth"],
                "n_affected": s["n_affected"],
                "n_defaulted": s["n_defaulted"],
                "total_stress": s["total_stress"],
                "avg_stress_pct": s["avg_stress"] * 100,
                "top_sectors": top_sector_text,
            }
        )
    rows.sort(key=lambda x: (-x["total_stress"], -x["cascade_depth"], x["ticker"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows, result_by_ticker


def build_compare_facts_html(
    rows: list[dict],
    date: str,
    threshold: float,
    regime: str,
    vix: float,
    shock_pct: int,
    model: str,
) -> str:
    """Build deterministic facts block for compare queries."""
    if not rows:
        return ""
    lines = [
        "<b>Simulation Facts (Deterministic, Compare)</b>",
        (
            f"• Date: {html.escape(date)} | Threshold: {threshold:.2f} | "
            f"Regime: {html.escape(regime)} (VIX {vix:.1f}) | "
            f"Shock: {shock_pct}% | Model: {html.escape(model)}"
        ),
    ]
    for row in rows:
        lines.append(
            "• "
            f"{html.escape(row['ticker'])}: waves {row['cascade_depth']} | "
            f"affected {row['n_affected']} | defaulted {row['n_defaulted']} | "
            f"total {row['total_stress']:.2f} | avg {row['avg_stress_pct']:.2f}% | "
            f"top sectors {html.escape(row['top_sectors'])}"
        )
    return "<br>".join(lines)


def build_compare_rows_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [
        "rank", "ticker", "cascade_depth", "n_affected", "n_defaulted",
        "total_stress", "avg_stress_pct", "top_sectors",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


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
    payload = {
        "q": query,
        "strategy": strategy,
        "deployment": primary_deployment,
        "date": parsed.get("date") if parsed else None,
        "ticker": parsed.get("ticker") if parsed else None,
        "tickers": parsed.get("tickers", []) if parsed else [],
        "shock": parsed.get("shock") if parsed else None,
        "threshold": threshold,
        "model": model,
        "risk_profile": risk_profile,
        "schema_version": schema_version,
    }
    return json.dumps(payload, sort_keys=True)


def find_cached_agent_response(
    *,
    exact_key: str,
    query: str,
    fingerprint: dict,
) -> tuple[dict | None, str]:
    cache = st.session_state.agent_response_cache
    exact = cache.get(exact_key)
    if exact:
        return exact, "exact"

    q_tokens = _tokenize_query(query)
    best_score = 0.0
    best_entry = None
    for entry in cache.values():
        if not isinstance(entry, dict):
            continue
        fp = entry.get("fingerprint", {})
        if not isinstance(fp, dict):
            continue
        if fp.get("tickers") != fingerprint.get("tickers"):
            continue
        if fp.get("shock") != fingerprint.get("shock"):
            continue
        if fp.get("date") != fingerprint.get("date"):
            continue
        if fp.get("model") != fingerprint.get("model"):
            continue
        if fp.get("risk_profile") != fingerprint.get("risk_profile"):
            continue
        e_tokens = set(entry.get("query_tokens", []))
        score = _jaccard_similarity(q_tokens, e_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= CACHE_SEMANTIC_MIN_SCORE:
        return best_entry, f"semantic:{best_score:.2f}"
    return None, "miss"


def get_agent_config_status() -> tuple[bool, str]:
    """Validate env vars needed by Agent Framework chat client."""
    try:
        settings = get_settings()
    except Exception as exc:
        return False, str(exc)

    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    missing = [k for k in required if not str(getattr(settings, k, "")).strip()]
    if missing:
        return False, f"Missing env vars: {', '.join(missing)}"
    return True, ""


def _run_async(coro):
    """Run coroutine from Streamlit sync context."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        # Streamlit normally runs sync, but keep a fallback for embedded loops.
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def _run_orchestrator_query_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    orchestrator = create_orchestrator(client)
    return await asyncio.wait_for(run_query(orchestrator, query), timeout=timeout_sec)


async def _run_simple_query_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    simple_agent = create_simple_agent(client)
    return await asyncio.wait_for(run_query(simple_agent, query), timeout=timeout_sec)


async def _run_parallel_workflow_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    return await asyncio.wait_for(run_parallel_workflow(client, query, timeout_sec=timeout_sec), timeout=timeout_sec)


def _run_direct_commentary_query(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    settings = get_settings()
    model_name = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT
    client = get_openai_client()
    resp = client.responses.create(
        model=model_name,
        input=query,
        timeout=timeout_sec,
    )
    return (resp.output_text or "").strip()


async def _run_critic_validation_async(
    *,
    query: str,
    facts_plain: str,
    candidate_json_text: str,
    timeout_sec: int = 12,
    deployment_name: str | None = None,
) -> dict:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    critic = create_critic_agent(client)
    prompt = (
        "Validate candidate JSON against deterministic evidence. Return strict JSON only.\n\n"
        f"User query:\n{query}\n\n"
        f"Deterministic evidence:\n{facts_plain}\n\n"
        f"Candidate JSON:\n{candidate_json_text}"
    )
    out = await asyncio.wait_for(run_query(critic, prompt), timeout=timeout_sec)
    parsed = extract_json_payload(out)
    if not isinstance(parsed, dict):
        return {
            "approved": False,
            "issues": ["Critic output was not valid JSON."],
            "required_fixes": ["Return strict JSON only."],
            "uncertainty_score": 0.8,
            "confidence_reason": "Critic parsing failed.",
        }
    return parsed


def run_critic_validation(
    *,
    query: str,
    facts_plain: str,
    candidate_json_text: str,
    timeout_sec: int = 12,
    deployment_name: str | None = None,
) -> dict:
    return _run_async(
        _run_critic_validation_async(
            query=query,
            facts_plain=facts_plain,
            candidate_json_text=candidate_json_text,
            timeout_sec=timeout_sec,
            deployment_name=deployment_name,
        )
    )


def run_agent_query(
    query: str,
    timeout_sec: int = 35,
    strategy: str = "simple",
    deployment_name: str | None = None,
) -> str:
    if strategy == "commentary_direct":
        return _run_direct_commentary_query(query, timeout_sec, deployment_name=deployment_name)
    if strategy == "orchestrator":
        return _run_async(_run_orchestrator_query_async(query, timeout_sec, deployment_name=deployment_name))
    if strategy == "workflow_parallel":
        return _run_async(_run_parallel_workflow_async(query, timeout_sec, deployment_name=deployment_name))
    return _run_async(_run_simple_query_async(query, timeout_sec, deployment_name=deployment_name))


def is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text


def is_retryable_gpt_error(exc: Exception) -> bool:
    if is_rate_limit_error(exc) or is_timeout_error(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sig in text for sig in ["connection", "temporarily unavailable", "service unavailable"])


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
    """Retry agent query on transient failures with bounded total wait."""
    attempt = 0
    started = time.perf_counter()
    while True:
        try:
            return run_agent_query(
                query=query,
                timeout_sec=timeout_sec,
                strategy=strategy,
                deployment_name=deployment_name,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            remaining = max_total_wait_sec - elapsed
            if (not is_retryable_gpt_error(exc)) or attempt >= max_retries or remaining <= 0.8:
                raise
            wait_sec = min(base_delay_sec * (2 ** attempt), max(0.5, remaining - 0.4))
            if on_backoff:
                on_backoff(wait_sec, attempt + 1, max_retries + 1)
            time.sleep(wait_sec)
            attempt += 1


def get_deployment_routing(high_quality_mode: bool) -> tuple[str, str]:
    """Return (primary, fallback) deployment names for the current run."""
    settings = get_settings()
    if high_quality_mode:
        return settings.AZURE_OPENAI_DEPLOYMENT, settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT
    return settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT, settings.AZURE_OPENAI_DEPLOYMENT


def run_gpt_diagnostic() -> str:
    """Run direct and agent-level GPT checks and return a text report."""
    lines = []
    settings = get_settings()
    lines.append(f"endpoint={settings.AZURE_OPENAI_ENDPOINT}")
    lines.append(f"deployment={settings.AZURE_OPENAI_DEPLOYMENT}")
    lines.append(f"fallback_deployment={settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT}")
    lines.append(f"api_version={settings.AZURE_OPENAI_API_VERSION}")

    # Direct SDK call (no tools, minimal path).
    t0 = time.perf_counter()
    try:
        client = get_openai_client()
        resp = client.responses.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            input="Reply with OK only.",
            timeout=20,
        )
        elapsed = time.perf_counter() - t0
        lines.append(f"direct_openai=OK ({elapsed:.2f}s) text={(resp.output_text or '').strip()[:80]}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        lines.append(f"direct_openai=ERR ({elapsed:.2f}s) {type(exc).__name__}: {exc}")

    # Agent Framework call (simple agent).
    t1 = time.perf_counter()
    try:
        txt = run_agent_query("Reply with OK only.", timeout_sec=30, strategy="simple")
        elapsed = time.perf_counter() - t1
        lines.append(f"agent_simple=OK ({elapsed:.2f}s) text={txt.strip()[:80]}")
    except Exception as exc:
        elapsed = time.perf_counter() - t1
        lines.append(f"agent_simple=ERR ({elapsed:.2f}s) {type(exc).__name__}: {exc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CORE: BUILD NETWORK
# ---------------------------------------------------------------------------
def do_build_network(date_str: str, threshold: float, emit_messages: bool = True):
    corr, actual_date = data_loader.get_correlation_matrix(date_str)
    G = network.build_network(corr, threshold=threshold, sector_dict=st.session_state.sector_dict)
    metrics = network.compute_global_metrics(G)

    regimes = data_loader.load_regime_data()
    ts = data_loader.find_nearest_date(date_str, regimes.index.tolist())
    regime_row = regimes.loc[ts]

    pos = _compute_layout(G)
    st.session_state.pos = pos
    st.session_state.graph_data = {
        "G": G, "date": str(actual_date.date()), "metrics": metrics,
        "regime": str(regime_row["Regime"]), "vix": float(regime_row["VIX"]),
        "threshold": threshold,
    }
    st.session_state.shock_result = None
    st.session_state.current_wave = -1

    if emit_messages:
        st.session_state.agent_messages.append(
            ("Architect", "🔧", "agent-architect",
             f"Network built for <b>{actual_date.date()}</b>: "
             f"{metrics['n_nodes']} nodes, {metrics['n_edges']:,} edges, "
             f"density {metrics['density']:.3f}. "
             f"Regime: <b>{regime_row['Regime']}</b> (VIX: {regime_row['VIX']:.1f}).")
        )
    return G


# ---------------------------------------------------------------------------
# CORE: RUN SHOCK
# ---------------------------------------------------------------------------
def do_run_shock(G: nx.Graph, ticker: str, shock_pct: int, model: str, emit_messages: bool = True):
    if ticker not in G:
        if emit_messages:
            st.session_state.agent_messages.append(
                ("Sentinel", "🛡️", "agent-sentinel",
                 f"⚠️ <b>{ticker}</b> not in network at this threshold. Try lower threshold.")
            )
        return

    result = contagion.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
    st.session_state.shock_result = result
    st.session_state.current_wave = result.cascade_depth

    sector_dict = st.session_state.sector_dict
    summary = result.summary()

    # Architect
    neighbors = network.get_node_neighbors(G, ticker)[:5]
    n_text = ", ".join(f"{t} (ρ={c:+.2f})" for t, c in neighbors)
    if emit_messages:
        st.session_state.agent_messages.append(
            ("Architect", "🔧", "agent-architect",
             f"<b>{ticker}</b> ({sector_dict.get(ticker, '?')}) — "
             f"{len(list(network.get_node_neighbors(G, ticker)))} connections. "
             f"Strongest: {n_text}.")
        )

    # Quant — severity tiers
    tiers = {"Critical >80%": 0, "High 50-80%": 0, "Moderate 20-50%": 0, "Low <20%": 0}
    for node, stress in result.node_stress.items():
        if node == ticker:
            continue
        if stress >= 0.8:
            tiers["Critical >80%"] += 1
        elif stress >= 0.5:
            tiers["High 50-80%"] += 1
        elif stress >= 0.2:
            tiers["Moderate 20-50%"] += 1
        elif stress > 0.01:
            tiers["Low <20%"] += 1
    tier_text = " · ".join(f"{k}: <b>{v}</b>" for k, v in tiers.items() if v > 0)

    if emit_messages:
        st.session_state.agent_messages.append(
            ("Quant", "📊", "agent-quant",
             f"<b>{model.replace('_', ' ').title()}</b> — "
             f"{ticker} at {shock_pct}% shock.<br>"
             f"→ Cascade: <b>{summary['cascade_depth']}</b> waves<br>"
             f"→ {tier_text}<br>"
             f"→ Total systemic stress: {summary['total_stress']:.1f} "
             f"(avg {summary['avg_stress']*100:.1f}%)")
        )

    # Advisor
    avg_stress = summary['avg_stress'] * 100
    risk_profile = st.session_state.get("risk_profile", "balanced")
    profile_hint = {
        "conservative": "Keep higher hedge ratio and reduce beta quickly.",
        "balanced": "Balance hedging with portfolio carry.",
        "aggressive": "Use tactical hedges and preserve selective upside.",
    }.get(risk_profile, "Balance hedging with portfolio carry.")
    if avg_stress > 30:
        risk_level, risk_class = "CRITICAL", "risk-critical"
        advice = (
            f"Systemic event. Avg stress {avg_stress:.1f}%. "
            f"<b>Act now:</b> (1) Broad hedges (SPY puts), "
            f"(2) Liquidate high-centrality names, (3) Cash up. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )
    elif avg_stress > 15:
        risk_level, risk_class = "HIGH", "risk-high"
        advice = (
            f"Severe contagion. Avg stress {avg_stress:.1f}%. "
            f"<b>Actions:</b> (1) Sector hedges, "
            f"(2) Review {ticker} counterparty exposure, (3) Tighten stops. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )
    elif avg_stress > 5:
        risk_level, risk_class = "ELEVATED", "risk-elevated"
        advice = (
            f"Moderate contagion. Avg stress {avg_stress:.1f}%. "
            f"<b>Monitor:</b> (1) VIX trajectory, "
            f"(2) Direct {ticker} exposure, (3) No broad hedging yet. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )
    else:
        risk_level, risk_class = "LOW", "risk-low"
        advice = (
            f"Contained. Avg stress {avg_stress:.1f}%. Minimal systemic impact. "
            f"Profile: <b>{risk_profile}</b> — {profile_hint}"
        )

    if emit_messages:
        st.session_state.agent_messages.append(
            ("Advisor", "📋", "agent-advisor",
             f'Risk: <span class="{risk_class}"><b>{risk_level}</b></span><br>{advice}')
        )


# ---------------------------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------------------------
def _compute_layout(G: nx.Graph) -> dict:
    try:
        return nx.kamada_kawai_layout(G)
    except Exception:
        return nx.spring_layout(G, k=0.4, iterations=80, seed=42)


def _stress_color(stress: float) -> str:
    if stress >= 0.8:
        return RISK_COLORS["critical"]
    elif stress >= 0.5:
        return RISK_COLORS["high"]
    elif stress >= 0.2:
        return RISK_COLORS["moderate"]
    elif stress > 0.01:
        return RISK_COLORS["low"]
    return "#334155"


def _build_base_layout(height: int = 580, title_text: str = "") -> go.Layout:
    """Shared Plotly layout for network graphs."""
    return go.Layout(
        showlegend=False, hovermode="closest",
        margin=dict(b=40, l=10, r=10, t=50),
        plot_bgcolor=PALETTE["bg_main"], paper_bgcolor=PALETTE["bg_main"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        title=dict(text=title_text, font=dict(color=PALETTE["accent_warm"], size=16),
                   x=0.5, xanchor="center") if title_text else None,
    )


def _bg_edge_trace(G: nx.Graph, pos: dict) -> go.Scatter:
    """Single trace with all edges as background (muted)."""
    ex, ey = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
    return go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.3, color=PLOT_EDGE_BG),
        hoverinfo="none", showlegend=False,
    )


def _stressed_edge_trace(G: nx.Graph, pos: dict, stress: dict) -> go.Scatter:
    """Single trace for edges where at least one endpoint is stressed."""
    ex, ey = [], []
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        ms = max(stress.get(u, 0), stress.get(v, 0))
        if ms > 0.01:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
    return go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=1.8, color=PLOT_EDGE_STRESS),
        hoverinfo="none", showlegend=False,
    )


def _node_trace(G: nx.Graph, pos: dict, stress: dict | None,
                shocked_node: str | None) -> go.Scatter:
    """Build the node scatter trace with stress coloring."""
    sector_dict = st.session_state.sector_dict
    valid = [n for n in G.nodes() if n in pos]
    nx_, ny_ = [pos[n][0] for n in valid], [pos[n][1] for n in valid]

    colors, sizes, texts, labels, outlines = [], [], [], [], []
    for node in valid:
        sector = sector_dict.get(node, "Unknown")
        s = stress.get(node, 0) if stress else 0
        is_s = node == shocked_node

        if stress:
            colors.append("#f8fafc" if is_s else _stress_color(s))
            sizes.append(28 if is_s else max(5, int(5 + s * 20)))
        else:
            colors.append(data_loader.SECTOR_COLORS.get(sector, RISK_COLORS["none"]))
            sizes.append(7)

        s_txt = f"<br>Stress: {s:.1%}" if stress else ""
        texts.append(f"<b>{node}</b><br>Sector: {sector}<br>Connections: {G.degree(node)}{s_txt}")
        labels.append(node if (is_s or (stress and s >= 0.5)) else "")
        outlines.append(RISK_COLORS["critical"] if is_s else PALETTE["bg_main"])

    return go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        hoverinfo="text", hovertext=texts,
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color=outlines)),
        text=labels, textposition="top center",
        textfont=dict(size=9, color=PALETTE["text_primary"]),
        showlegend=False,
    )


def build_graph_figure(G: nx.Graph, pos: dict) -> go.Figure:
    """Static network figure (no shock — sector-colored)."""
    return go.Figure(
        data=[_bg_edge_trace(G, pos), _node_trace(G, pos, None, None)],
        layout=_build_base_layout(),
    )


def build_animated_figure(
    G: nx.Graph, pos: dict, result, blast_radius_only: bool = False,
) -> go.Figure:
    """Animated figure with Plotly native frames — one frame per cascade wave.

    Uses 3 traces per frame:
      0 — background edges (static)
      1 — stressed edges (changes per wave)
      2 — nodes (colors/sizes change per wave)
    Play/Pause/Slider built into the Plotly chart.
    """
    sector_dict = st.session_state.sector_dict
    shocked_node = result.shocked_node
    n_waves = result.cascade_depth

    # Optional blast-radius subgraph
    if blast_radius_only:
        visible = {n for n, s in result.node_stress.items() if s > 0.01}
        visible.add(shocked_node)
        for n in list(visible):
            for nbr in G.neighbors(n):
                visible.add(nbr)
        G = G.subgraph(visible).copy()

    bg = _bg_edge_trace(G, pos)
    valid = [n for n in G.nodes() if n in pos]

    # Pre-compute stress dict per wave
    def _wave_stress(w: int) -> dict:
        s = {n: 0.0 for n in G.nodes()}
        s[shocked_node] = result.shock_magnitude
        for wn, nodes in result.cascade_waves:
            if wn <= w:
                for node in nodes:
                    s[node] = result.node_stress[node]
        return s

    # Build frames
    frames = []
    for w in range(n_waves + 1):
        ws = _wave_stress(w)
        label = f"Wave {w}/{n_waves}" if w > 0 else "Initial Shock"
        frames.append(go.Frame(
            data=[
                bg,
                _stressed_edge_trace(G, pos, ws),
                _node_trace(G, pos, ws, shocked_node),
            ],
            name=str(w),
            layout=go.Layout(title=dict(text=label, font=dict(color=PALETTE["accent_warm"], size=16),
                                        x=0.5, xanchor="center")),
        ))

    # Initial state = wave 0
    fig = go.Figure(data=frames[0].data, frames=frames, layout=_build_base_layout(
        height=580, title_text="Initial Shock"))

    # Slider steps
    steps = []
    for w in range(n_waves + 1):
        steps.append(dict(
            args=[[str(w)], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                                 transition=dict(duration=300))],
            label=f"W{w}" if w > 0 else "💥",
            method="animate",
        ))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.08, y=1.12, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(
                         frame=dict(duration=1200, redraw=True),
                         transition=dict(duration=400, easing="cubic-in-out"),
                         fromcurrent=True, mode="immediate",
                     )]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode="immediate",
                     )]),
            ],
        )],
        sliders=[dict(
            active=0, steps=steps,
            x=0.1, len=0.8, y=-0.02,
            currentvalue=dict(prefix="Cascade: ", visible=True,
                              font=dict(color=PALETTE["accent_warm"], size=13)),
            tickcolor=PALETTE["text_muted"], font=dict(color=PALETTE["text_muted"]),
            bgcolor=PALETTE["surface_1"], bordercolor=PALETTE["surface_1"],
            activebgcolor=PALETTE["accent_warm"],
        )],
    )

    return fig


def build_severity_df(result: contagion.ShockResult) -> pd.DataFrame:
    sector_dict = st.session_state.sector_dict
    sector_data = {}
    for node, stress in result.node_stress.items():
        sector = sector_dict.get(node, "Unknown")
        if sector not in sector_data:
            sector_data[sector] = {"stresses": [], "defaulted": 0}
        sector_data[sector]["stresses"].append(stress)
        if stress >= 1.0:
            sector_data[sector]["defaulted"] += 1
    rows = []
    for sector, d in sector_data.items():
        avg = np.mean(d["stresses"])
        hit = sum(1 for s in d["stresses"] if s > 0.01)
        rows.append({
            "Sector": sector, "Avg Stress %": round(avg * 100, 1),
            "Nodes Hit": hit, "Total": len(d["stresses"]), "Defaulted": d["defaulted"],
        })
    return pd.DataFrame(rows).sort_values("Avg Stress %", ascending=False)


def compute_systemic_risk_index(result: contagion.ShockResult, total_nodes: int) -> tuple[float, str]:
    """Compute a compact 0-100 systemic risk index for dashboard gauge."""
    summary = result.summary()
    n_nodes = max(1, int(total_nodes))
    affected_pct = (summary["n_affected"] / n_nodes) * 100.0
    defaulted_pct = (summary["n_defaulted"] / n_nodes) * 100.0
    avg_stress_pct = float(summary["avg_stress"]) * 100.0
    depth_score = min(100.0, float(summary["cascade_depth"]) * 12.5)

    score = (
        0.40 * avg_stress_pct
        + 0.25 * affected_pct
        + 0.20 * depth_score
        + 0.15 * defaulted_pct
    )
    score = max(0.0, min(100.0, score))

    if score >= 70:
        label = "CRITICAL"
    elif score >= 50:
        label = "HIGH"
    elif score >= 30:
        label = "ELEVATED"
    else:
        label = "LOW"
    return score, label


def build_sector_impact_bar_figure(result: contagion.ShockResult, top_n: int = 10) -> go.Figure:
    """Bar chart: affected nodes by sector, colored by average stress."""
    df = build_severity_df(result).copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values(["Nodes Hit", "Avg Stress %"], ascending=[False, False]).head(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Sector"],
            y=df["Nodes Hit"],
            marker=dict(
                color=df["Avg Stress %"],
                colorscale=PLOTLY_STRESS_COLORSCALE,
                colorbar=dict(title="Avg Stress %"),
            ),
            customdata=np.stack([df["Avg Stress %"], df["Defaulted"]], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Nodes Hit: %{y}<br>"
                "Avg Stress: %{customdata[0]:.1f}%<br>"
                "Defaulted: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Affected Nodes by Sector",
        height=320,
        margin=dict(l=30, r=20, t=40, b=30),
        plot_bgcolor=PALETTE["bg_main"],
        paper_bgcolor=PALETTE["bg_main"],
        font=dict(color=PALETTE["text_primary"]),
        xaxis=dict(color=PALETTE["text_muted"], tickangle=-20),
        yaxis=dict(color=PALETTE["text_muted"], showgrid=True, gridcolor=PALETTE["surface_1"]),
    )
    return fig


def build_stress_tier_donut_figure(result: contagion.ShockResult) -> go.Figure:
    """Donut chart: distribution of stressed nodes by severity tier."""
    tiers = {"Critical >80%": 0, "High 50-80%": 0, "Moderate 20-50%": 0, "Low 1-20%": 0}
    for node, stress in result.node_stress.items():
        if node == result.shocked_node or stress <= 0.01:
            continue
        if stress >= 0.8:
            tiers["Critical >80%"] += 1
        elif stress >= 0.5:
            tiers["High 50-80%"] += 1
        elif stress >= 0.2:
            tiers["Moderate 20-50%"] += 1
        else:
            tiers["Low 1-20%"] += 1

    labels = [k for k, v in tiers.items() if v > 0]
    values = [v for _, v in tiers.items() if v > 0]
    if not labels:
        labels = ["No stressed nodes"]
        values = [1]

    colors = {
        "Critical >80%": RISK_COLORS["critical"],
        "High 50-80%": RISK_COLORS["high"],
        "Moderate 20-50%": RISK_COLORS["moderate"],
        "Low 1-20%": RISK_COLORS["low"],
        "No stressed nodes": RISK_COLORS["none"],
    }
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=[colors.get(lbl, RISK_COLORS["none"]) for lbl in labels]),
                sort=False,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Stress Tier Distribution",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=PALETTE["bg_main"],
        paper_bgcolor=PALETTE["bg_main"],
        font=dict(color=PALETTE["text_primary"]),
        legend=dict(font=dict(color=PALETTE["text_muted"], size=10)),
    )
    return fig


def build_systemic_risk_gauge_figure(result: contagion.ShockResult, total_nodes: int) -> tuple[go.Figure, float, str]:
    """Gauge chart for systemic risk index + label."""
    score, label = compute_systemic_risk_index(result, total_nodes)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": PALETTE["text_muted"]},
                "bar": {"color": PALETTE["accent_warm"]},
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 50], "color": "#713f12"},
                    {"range": [50, 70], "color": "#7c2d12"},
                    {"range": [70, 100], "color": "#7f1d1d"},
                ],
            },
        )
    )
    fig.update_layout(
        title=f"Systemic Risk Index ({label})",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor=PALETTE["bg_main"],
        paper_bgcolor=PALETTE["bg_main"],
        font=dict(color=PALETTE["text_primary"]),
    )
    return fig, score, label


def build_wave_trend_figure(result: contagion.ShockResult) -> go.Figure:
    """Combo chart: nodes hit per wave + wave stress contribution."""
    rows = [
        {"Wave": 0, "Nodes Hit": 1, "Wave Stress %": round(result.shock_magnitude * 100, 2)},
    ]
    for wave, nodes in result.cascade_waves:
        wave_stress = sum(result.node_stress.get(n, 0.0) for n in nodes)
        rows.append(
            {
                "Wave": int(wave),
                "Nodes Hit": int(len(nodes)),
                "Wave Stress %": round(float(wave_stress) * 100, 2),
            }
        )
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["Wave"],
            y=df["Nodes Hit"],
            name="Nodes Hit",
            marker_color=PALETTE["accent_cool"],
            opacity=0.85,
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Wave"],
            y=df["Wave Stress %"],
            name="Wave Stress %",
            mode="lines+markers",
            line=dict(color=PALETTE["accent_warm"], width=2),
            marker=dict(size=6),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Cascade Wave Dynamics",
        height=320,
        margin=dict(l=35, r=35, t=40, b=30),
        plot_bgcolor=PALETTE["bg_main"],
        paper_bgcolor=PALETTE["bg_main"],
        font=dict(color=PALETTE["text_primary"]),
        xaxis=dict(title="Wave", color=PALETTE["text_muted"], dtick=1),
        yaxis=dict(title="Nodes Hit", color=PALETTE["accent_cool"], showgrid=True, gridcolor=PALETTE["surface_1"]),
        yaxis2=dict(
            title="Wave Stress %",
            overlaying="y",
            side="right",
            color=PALETTE["accent_warm"],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=PALETTE["text_muted"], size=10)),
    )
    return fig


def build_timeline_figure() -> go.Figure | None:
    """Mini timeline chart: network density + VIX over time."""
    try:
        nm = data_loader.load_network_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nm.index, y=nm["density"], name="Density",
            line=dict(color=PALETTE["accent_cool"], width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=nm.index, y=nm["vix"] / 100, name="VIX / 100",
            line=dict(color=PALETTE["accent_warm"], width=1.5),
            yaxis="y2",
        ))

        # Mark crises
        for name, (start, end) in data_loader.CRISIS_EVENTS.items():
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=PLOT_EVENT_FILL, line_width=0,
                annotation_text=name.split(" ")[0],
                annotation_position="top left",
                annotation_font_size=8,
                annotation_font_color=PALETTE["accent_warm"],
            )

        # Mark selected date
        if st.session_state.graph_data:
            fig.add_vline(
                x=st.session_state.graph_data["date"],
                line_dash="dash", line_color=PALETTE["text_primary"], line_width=1,
            )

        fig.update_layout(
            height=200, margin=dict(l=40, r=40, t=20, b=30),
            plot_bgcolor=PALETTE["bg_main"], paper_bgcolor=PALETTE["bg_main"],
            legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(size=10, color=PALETTE["text_muted"])),
            xaxis=dict(showgrid=False, color=PALETTE["text_muted"]),
            yaxis=dict(title="Density", showgrid=True, gridcolor=PALETTE["surface_1"], color=PALETTE["accent_cool"], range=[0, 0.8]),
            yaxis2=dict(title="VIX/100", overlaying="y", side="right", showgrid=False,
                        color=PALETTE["accent_warm"], range=[0, 0.8]),
        )
        return fig
    except Exception:
        return None


def agent_message(name: str, icon: str, css_class: str, text: str):
    """Render a styled agent message."""
    st.markdown(
        f'<div class="agent-msg {css_class}">'
        f'<b>{icon} {name}</b><br>{text}</div>',
        unsafe_allow_html=True,
    )


def generate_report_text() -> str:
    """Generate a text report for download."""
    gd = st.session_state.graph_data
    sr = st.session_state.shock_result
    if not gd or not sr:
        return "No simulation results to report."

    summary = sr.summary()
    lines = [
        "=" * 60,
        "RISKSENTINEL — SYSTEMIC RISK ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Date: {gd['date']}",
        f"Market Regime: {gd['regime']} (VIX: {gd['vix']:.1f})",
        f"Network: {gd['metrics']['n_nodes']} nodes, {gd['metrics']['n_edges']} edges, "
        f"density {gd['metrics']['density']:.3f}",
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
    sector_dict = st.session_state.sector_dict
    for item in summary['top_10_affected']:
        t = item['ticker']
        lines.append(f"  {t:6s} ({sector_dict.get(t, '?'):30s}) stress={item['stress']*100:.1f}%")

    lines.extend(["", "--- SECTOR BREAKDOWN ---"])
    sev_df = build_severity_df(sr)
    for _, row in sev_df.iterrows():
        lines.append(f"  {row['Sector']:30s} avg={row['Avg Stress %']:5.1f}%  hit={row['Nodes Hit']}/{row['Total']}")

    lines.extend([
        "",
        "--- AGENT MESSAGES ---",
    ])
    for name, icon, css, text in st.session_state.agent_messages:
        clean = re.sub(r'<[^>]+>', '', text)
        lines.append(f"[{name}] {clean}")

    lines.extend(["", "=" * 60,
                   "Generated by RiskSentinel — Microsoft AI Dev Days Hackathon 2026",
                   "=" * 60])
    return "\n".join(lines)


def generate_report_markdown() -> str:
    """Executive markdown brief for judges and stakeholders."""
    gd = st.session_state.graph_data
    sr = st.session_state.shock_result
    if not gd or not sr:
        return "# RiskSentinel Brief\n\nNo simulation results available."

    summary = sr.summary()
    rm = st.session_state.last_run_metrics or {}
    lines = [
        "# RiskSentinel Executive Brief",
        "",
        f"- **Date**: {gd['date']}",
        f"- **Regime**: {gd['regime']} (VIX {gd['vix']:.1f})",
        (
            f"- **Scenario**: {summary['shocked_node']} shock {summary['shock_magnitude']*100:.0f}% "
            f"with {summary['model']}"
        ),
        (
            f"- **Impact**: {summary['n_affected']} affected, {summary['n_defaulted']} defaulted, "
            f"{summary['cascade_depth']} waves, avg stress {summary['avg_stress']*100:.2f}%"
        ),
        (
            f"- **Runtime**: total {rm.get('total_sec', 0.0):.1f}s, "
            f"local {rm.get('local_sec', 0.0) if isinstance(rm.get('local_sec'), float) else 0.0:.1f}s, "
            f"gpt {rm.get('gpt_sec', 0.0) if isinstance(rm.get('gpt_sec'), float) else 0.0:.1f}s"
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


def generate_action_pack_ceo_brief() -> str:
    """One-page CEO narrative for fast executive decisioning."""
    shock_summary = st.session_state.shock_result.summary() if st.session_state.shock_result else None
    return reporting.generate_action_pack_ceo_brief(
        graph_data=st.session_state.graph_data,
        shock_summary=shock_summary,
        commander=st.session_state.commander_results or {},
        autonomous=st.session_state.autonomous_results or {},
        portfolio=st.session_state.portfolio_copilot or {},
    )


def generate_action_pack_runbook() -> str:
    """Operational runbook for risk desk execution."""
    return reporting.generate_action_pack_runbook(
        commander=st.session_state.commander_results or {},
        portfolio=st.session_state.portfolio_copilot or {},
    )


def generate_action_pack_machine_json() -> str:
    """Machine-readable action payload for automation / MCP style handoff."""
    payload = reporting.build_action_pack_payload(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        market_context=st.session_state.graph_data,
        commander=st.session_state.commander_results,
        autonomous_stress_test=st.session_state.autonomous_results,
        portfolio_copilot=st.session_state.portfolio_copilot,
        trace_summary=st.session_state.run_trace,
        policy_plan=st.session_state.latest_policy_plan,
        executor_log=st.session_state.latest_executor_log,
        session_memory=st.session_state.session_decisions[-20:],
    )
    return reporting.generate_action_pack_machine_json(payload)


def generate_trace_bundle_json() -> str:
    """Downloadable explainability payload for audit/demo."""
    quality_summary = summarize_quality(st.session_state.run_trace_history)
    judge_kpis = build_judge_kpis(st.session_state.run_trace_history)
    payload = {
        "last_run_metrics": st.session_state.last_run_metrics,
        "trace": st.session_state.run_trace,
        "history_size": len(st.session_state.run_trace_history),
        "quality_summary": quality_summary,
        "judge_kpis": judge_kpis,
        "rag_last_docs": st.session_state.rag_last_docs,
        "risk_profile": st.session_state.risk_profile,
        "policy_plan": st.session_state.latest_policy_plan,
        "executor_log": st.session_state.latest_executor_log,
        "session_decisions": st.session_state.session_decisions[-20:],
        "scenario_commander": st.session_state.commander_results,
        "autonomous_stress_test": st.session_state.autonomous_results,
        "portfolio_copilot": st.session_state.portfolio_copilot,
    }
    return json.dumps(reporting.json_safe(payload), indent=2)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ RiskSentinel")
    st.caption("Agentic Systemic Risk Simulator")
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
    scenario_names = [s["name"] for s in SCENARIO_PACK]
    st.session_state.scenario_pack_choice = st.selectbox(
        "Judge scenario",
        options=scenario_names,
        index=scenario_names.index(st.session_state.scenario_pack_choice)
        if st.session_state.scenario_pack_choice in scenario_names else 0,
    )
    selected_scenario = next(s for s in SCENARIO_PACK if s["name"] == st.session_state.scenario_pack_choice)
    st.caption(selected_scenario["query"])
    st.caption(f"Expected route: {selected_scenario['expected_route']}")
    if st.button("▶ Run Scenario", use_container_width=True):
        st.session_state.pending_chat_query = selected_scenario["query"]
        st.rerun()

    st.divider()

    # === CRISIS PRESETS ===
    st.markdown("### ⚡ Crisis Presets")
    preset_cols = st.columns(2)
    for i, (name, params) in enumerate(CRISIS_PRESETS.items()):
        col = preset_cols[i % 2]
        if col.button(name, key=f"preset_{name}", use_container_width=True):
            st.session_state.sel_date = params["date"]
            st.session_state.sel_ticker = params["ticker"]
            st.session_state.sel_shock = params["shock"]
            st.session_state.sel_threshold = params["threshold"]
            st.session_state.agent_messages = []
            # Build + shock
            G = do_build_network(params["date"], params["threshold"])
            do_run_shock(G, params["ticker"], params["shock"], "debtrank")
            st.rerun()

    st.divider()

    # === MANUAL CONTROLS ===
    st.markdown("### 📅 Network Date")
    available_dates = data_loader.get_available_dates()
    date_strings = [str(d.date()) for d in available_dates]
    init_date = st.session_state.sel_date or date_strings[-1]
    if init_date not in date_strings:
        init_date = min(date_strings, key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(init_date)))
    selected_date = st.select_slider("Date", options=date_strings, value=init_date, label_visibility="collapsed")

    st.divider()
    st.markdown("### 💥 Shock Scenario")

    shocked_ticker = st.selectbox(
        "Target stock", options=st.session_state.tickers,
        index=st.session_state.tickers.index(st.session_state.sel_ticker)
        if st.session_state.sel_ticker in st.session_state.tickers else 0,
    )
    shock_pct = st.slider("Shock %", 10, 100, st.session_state.sel_shock, 10)
    shock_model = st.selectbox("Model", ["debtrank", "linear_threshold", "cascade_removal"], index=0)
    threshold = st.slider("Corr. threshold", 0.2, 0.8, st.session_state.sel_threshold, 0.05,
                          help="Higher = sparser = more realistic contagion")

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
    st.caption("Advanced GPT controls, diagnostics, and explainability tools are in the `Settings` tab.")


# ---------------------------------------------------------------------------
# BUILD / SHOCK ACTIONS
# ---------------------------------------------------------------------------
sector_dict_ctx = dict(st.session_state.get("sector_dict") or {})
tickers_ctx = list(st.session_state.get("tickers") or [])
risk_profile_ctx = str(st.session_state.get("risk_profile", "balanced"))
auto_portfolio_n_ctx = int(st.session_state.get("auto_portfolio_n", 5) or 5)
portfolio_text_ctx = str(st.session_state.get("portfolio_text", "") or "")
agentic_requested = bool(commander_btn or autonomous_btn or auto_portfolio_btn or portfolio_btn or full_demo_btn)
if agentic_requested and (not sector_dict_ctx or not tickers_ctx):
    try:
        sector_dict_ctx = dict(data_loader.get_sector_dict() or {})
        tickers_ctx = list(data_loader.get_ticker_list() or [])
        st.session_state.sector_dict = sector_dict_ctx
        st.session_state.tickers = tickers_ctx
    except Exception as exc:
        st.session_state.last_agentic_action = (
            "Agentic ops unavailable: reference data not loaded "
            f"({type(exc).__name__}: {exc})."
        )

if build_btn or (AUTO_BUILD_ON_START and st.session_state.graph_data is None):
    st.session_state.agent_messages = []
    do_build_network(selected_date, threshold)

if shock_btn and st.session_state.graph_data:
    st.session_state.comparison = None
    do_run_shock(st.session_state.graph_data["G"], shocked_ticker, shock_pct, shock_model)

if compare_btn and st.session_state.graph_data:
    G = st.session_state.graph_data["G"]
    if shocked_ticker in G:
        st.session_state.comparison = contagion.compare_models(G, shocked_ticker, shock_pct / 100.0)
        # Also set shock_result to debtrank for the main view
        st.session_state.shock_result = st.session_state.comparison["debtrank"]
        st.session_state.current_wave = st.session_state.shock_result.cascade_depth
        st.session_state.agent_messages = [
            ("Quant", "📊", "agent-quant",
             f"<b>Model Comparison</b> — {shocked_ticker} at {shock_pct}% shock. "
             f"See comparison table below the network graph.")
        ]

if commander_btn and sector_dict_ctx:
    cmd_key = _agentic_cache_key(
        "scenario_commander",
        date=selected_date,
        threshold=round(float(threshold), 3),
        shock_pct=int(shock_pct),
        model=shock_model,
        top_n=DEFAULT_COMMANDER_TOP_N,
    )
    with st.spinner("Running Scenario Commander..."):
        cmd_res, from_cache = _run_agentic_operation(
            op_name="Scenario Commander",
            cache_key=cmd_key,
            fn=lambda: run_scenario_commander(
                date_str=selected_date,
                threshold=threshold,
                shock_pct=shock_pct,
                model=shock_model,
                top_n=DEFAULT_COMMANDER_TOP_N,
                sector_dict=sector_dict_ctx,
            ),
        )
    if not cmd_res.get("ok", True):
        st.session_state.last_agentic_action = f"Scenario Commander failed: {cmd_res.get('error', 'unknown error')}."
    else:
        st.session_state.commander_results = cmd_res
        top = cmd_res.get("top_pick")
        if top:
            cache_suffix = " (cached)" if from_cache else ""
            st.session_state.agent_messages.append(
                (
                    "Sentinel",
                    "🛡️",
                    "agent-sentinel",
                    f"Scenario Commander completed{cache_suffix}. Top systemic seed: <b>{top['ticker']}</b> "
                    f"(score {top['risk_score']:.1f}, depth {top['cascade_depth']}).",
                )
            )
            st.session_state.last_agentic_action = (
                f"Scenario Commander done{cache_suffix}: top={top['ticker']} score={top['risk_score']:.1f}. "
                "Open Dashboard tab for full ranking."
            )

if autonomous_btn and sector_dict_ctx:
    auto_key = _agentic_cache_key(
        "autonomous_stress_test",
        date=selected_date,
        threshold=round(float(threshold), 3),
        model=shock_model,
        shock_grid=AUTONOMOUS_SHOCK_GRID,
        max_seeds=DEFAULT_AUTONOMOUS_SEEDS,
    )
    with st.spinner("Running Autonomous Stress Test..."):
        auto_res, from_cache = _run_agentic_operation(
            op_name="Autonomous Stress Test",
            cache_key=auto_key,
            fn=lambda: run_autonomous_stress_test(
                date_str=selected_date,
                threshold=threshold,
                model=shock_model,
                sector_dict=sector_dict_ctx,
            ),
        )
    if not auto_res.get("ok", True):
        st.session_state.last_agentic_action = f"Auto Stress failed: {auto_res.get('error', 'unknown error')}."
    else:
        st.session_state.autonomous_results = auto_res
        rows = auto_res.get("rows") or []
        if rows:
            lead = rows[0]
            cache_suffix = " (cached)" if from_cache else ""
            st.session_state.agent_messages.append(
                (
                    "Sentinel",
                    "🛡️",
                    "agent-sentinel",
                    f"Autonomous Stress Test completed{cache_suffix}. Lead fragility: <b>{lead['ticker']}</b> @ "
                    f"{lead['shock_pct']}% (score {lead['risk_score']:.1f}).",
                )
            )
            st.session_state.last_agentic_action = (
                f"Auto Stress done{cache_suffix}: lead={lead['ticker']} shock={lead['shock_pct']}% "
                f"score={lead['risk_score']:.1f}. Open Dashboard tab for table."
            )

if auto_portfolio_btn and sector_dict_ctx:
    auto_port_key = _agentic_cache_key(
        "auto_portfolio",
        date=selected_date,
        threshold=round(float(threshold), 3),
        n_positions=int(st.session_state.auto_portfolio_n),
    )
    with st.spinner("Generating portfolio from current network..."):
        auto_pack, from_cache = _run_agentic_operation(
            op_name="Auto Portfolio",
            cache_key=auto_port_key,
            fn=lambda: build_auto_portfolio_from_network(
                date_str=selected_date,
                threshold=threshold,
                n_positions=auto_portfolio_n_ctx,
                sector_dict=sector_dict_ctx,
            ),
        )
    if auto_pack.get("ok"):
        st.session_state.portfolio_text = auto_pack.get("portfolio_text", "")
        top = (auto_pack.get("rows") or [{}])[0]
        cache_suffix = " (cached)" if from_cache else ""
        st.session_state.last_agentic_action = (
            f"Auto-portfolio ready{cache_suffix} ({len(auto_pack.get('rows', []))} positions) "
            f"for {auto_pack.get('date')}, regime {auto_pack.get('regime')}."
        )
        st.session_state.agent_messages.append(
            (
                "Architect",
                "🔧",
                "agent-architect",
                "Auto portfolio created from network topology "
                f"(method: {auto_pack.get('method')}{cache_suffix}). Top ticker: <b>{top.get('ticker', 'n/a')}</b>.",
            )
        )
    else:
        st.session_state.last_agentic_action = (
            f"Auto-portfolio failed: {auto_pack.get('error', 'unknown error')}."
        )

if portfolio_btn and sector_dict_ctx and tickers_ctx:
    portfolio_hash = hashlib.sha1(portfolio_text_ctx.strip().encode("utf-8")).hexdigest()
    copilot_key = _agentic_cache_key(
        "portfolio_copilot",
        portfolio_hash=portfolio_hash,
        date=selected_date,
        threshold=round(float(threshold), 3),
        model=shock_model,
        shock_pct=int(shock_pct),
        risk_profile=st.session_state.risk_profile,
    )
    with st.spinner("Running Portfolio Co-Pilot..."):
        cop, from_cache = _run_agentic_operation(
            op_name="Portfolio Co-Pilot",
            cache_key=copilot_key,
            fn=lambda: run_portfolio_copilot(
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
        st.session_state.portfolio_copilot = cop
    if cop.get("ok"):
        cache_suffix = " (cached)" if from_cache else ""
        st.session_state.agent_messages.append(
            (
                "Advisor",
                "📋",
                "agent-advisor",
                f"Portfolio Co-Pilot ready{cache_suffix}. Expected stress: "
                f"<b>{cop.get('expected_stress_pct', 0.0):.1f}%</b> "
                f"→ after hedges <b>{cop.get('expected_stress_pct_after_hedge', 0.0):.1f}%</b>.",
            )
        )
        st.session_state.last_agentic_action = (
            f"Portfolio Co-Pilot done{cache_suffix}: stress {cop.get('expected_stress_pct', 0.0):.1f}% "
            f"-> {cop.get('expected_stress_pct_after_hedge', 0.0):.1f}%."
        )
    elif cop.get("timeout"):
        st.session_state.last_agentic_action = f"Portfolio Co-Pilot timeout: {cop.get('error', 'try again')}."
    elif cop.get("error"):
        st.session_state.last_agentic_action = f"Portfolio Co-Pilot failed: {cop.get('error')}."
    else:
        errs = (cop.get("errors") or [])[:3]
        st.session_state.last_agentic_action = (
            "Portfolio Co-Pilot input invalid. "
            + (" | ".join(errs) if errs else "Check ticker,weight format.")
        )

if full_demo_btn and sector_dict_ctx and tickers_ctx:
    run = agentic_ops.build_full_demo_steps(now_utc=datetime.now(timezone.utc).isoformat())
    failed = False
    with st.spinner("Running Full Agentic Demo..."):
        try:
            do_build_network(selected_date, threshold)
            agentic_ops.append_demo_step(run, "Build network", "ok", f"date={selected_date} threshold={threshold:.2f}")
        except Exception as exc:
            failed = True
            agentic_ops.append_demo_step(run, "Build network", "failed", f"{type(exc).__name__}: {exc}")

        if not failed:
            cmd_key = _agentic_cache_key(
                "scenario_commander",
                date=selected_date,
                threshold=round(float(threshold), 3),
                shock_pct=int(shock_pct),
                model=shock_model,
                top_n=DEFAULT_COMMANDER_TOP_N,
            )
            cmd_res, cmd_cached = _run_agentic_operation(
                op_name="Scenario Commander",
                cache_key=cmd_key,
                fn=lambda: run_scenario_commander(
                    date_str=selected_date,
                    threshold=threshold,
                    shock_pct=shock_pct,
                    model=shock_model,
                    top_n=DEFAULT_COMMANDER_TOP_N,
                    sector_dict=sector_dict_ctx,
                ),
            )
            if cmd_res.get("ok", True):
                st.session_state.commander_results = cmd_res
                top = (cmd_res.get("top_pick") or {}).get("ticker", "n/a")
                agentic_ops.append_demo_step(
                    run, "Scenario Commander", "ok", f"top={top} cached={cmd_cached}"
                )
            else:
                failed = True
                agentic_ops.append_demo_step(
                    run, "Scenario Commander", "failed", cmd_res.get("error", "unknown error")
                )

        if not failed:
            auto_key = _agentic_cache_key(
                "autonomous_stress_test",
                date=selected_date,
                threshold=round(float(threshold), 3),
                model=shock_model,
                shock_grid=AUTONOMOUS_SHOCK_GRID,
                max_seeds=DEFAULT_AUTONOMOUS_SEEDS,
            )
            auto_res, auto_cached = _run_agentic_operation(
                op_name="Autonomous Stress Test",
                cache_key=auto_key,
                fn=lambda: run_autonomous_stress_test(
                    date_str=selected_date,
                    threshold=threshold,
                    model=shock_model,
                    sector_dict=sector_dict_ctx,
                ),
            )
            if auto_res.get("ok", True):
                st.session_state.autonomous_results = auto_res
                lead = ((auto_res.get("rows") or [{}])[0]).get("ticker", "n/a")
                agentic_ops.append_demo_step(
                    run, "Autonomous Stress Test", "ok", f"lead={lead} cached={auto_cached}"
                )
            else:
                failed = True
                agentic_ops.append_demo_step(
                    run, "Autonomous Stress Test", "failed", auto_res.get("error", "unknown error")
                )

        if not failed and not portfolio_text_ctx.strip():
            auto_port_key = _agentic_cache_key(
                "auto_portfolio",
                date=selected_date,
                threshold=round(float(threshold), 3),
                n_positions=int(st.session_state.auto_portfolio_n),
            )
            auto_pack, p_cached = _run_agentic_operation(
                op_name="Auto Portfolio",
                cache_key=auto_port_key,
                fn=lambda: build_auto_portfolio_from_network(
                    date_str=selected_date,
                    threshold=threshold,
                    n_positions=auto_portfolio_n_ctx,
                    sector_dict=sector_dict_ctx,
                ),
            )
            if auto_pack.get("ok"):
                portfolio_text_ctx = str(auto_pack.get("portfolio_text", "") or "")
                st.session_state.portfolio_text = portfolio_text_ctx
                agentic_ops.append_demo_step(
                    run,
                    "Auto Portfolio",
                    "ok",
                    f"positions={len(auto_pack.get('rows', []))} cached={p_cached}",
                )
            else:
                failed = True
                agentic_ops.append_demo_step(
                    run, "Auto Portfolio", "failed", auto_pack.get("error", "unknown error")
                )

        if not failed:
            portfolio_hash = hashlib.sha1(portfolio_text_ctx.strip().encode("utf-8")).hexdigest()
            copilot_key = _agentic_cache_key(
                "portfolio_copilot",
                portfolio_hash=portfolio_hash,
                date=selected_date,
                threshold=round(float(threshold), 3),
                model=shock_model,
                shock_pct=int(shock_pct),
                risk_profile=st.session_state.risk_profile,
            )
            cop, cop_cached = _run_agentic_operation(
                op_name="Portfolio Co-Pilot",
                cache_key=copilot_key,
                fn=lambda: run_portfolio_copilot(
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
            st.session_state.portfolio_copilot = cop
            if cop.get("ok"):
                agentic_ops.append_demo_step(
                    run,
                    "Portfolio Co-Pilot",
                    "ok",
                    (
                        f"stress={cop.get('expected_stress_pct', 0.0):.1f}% "
                        f"avoided={cop.get('estimated_loss_avoided_pct', 0.0):.1f}% cached={cop_cached}"
                    ),
                )
            else:
                failed = True
                detail = cop.get("error") or " | ".join((cop.get("errors") or [])[:2]) or "invalid input"
                agentic_ops.append_demo_step(run, "Portfolio Co-Pilot", "failed", detail)

    run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if failed:
        has_ok = any(s.get("status") == "ok" for s in run.get("steps", []))
        run["status"] = "partial" if has_ok else "failed"
    else:
        run["status"] = "completed"
    st.session_state.full_demo_last_run = run
    if failed:
        failed_steps = [s["step"] for s in run.get("steps", []) if s.get("status") == "failed"]
        st.session_state.last_agentic_action = (
            "Full Agentic Demo partial/failed. "
            + (f"Failed step(s): {', '.join(failed_steps)}." if failed_steps else "Check logs.")
        )
    else:
        st.session_state.last_agentic_action = "Full Agentic Demo completed (Build + Commander + Auto Stress + Co-Pilot)."
        st.session_state.agent_messages.append(
            (
                "Sentinel",
                "🛡️",
                "agent-sentinel",
                "Full Agentic Demo completed. Review Dashboard and Explainability tabs for ranked vulnerabilities and audit trail.",
            )
        )


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    '<h1 style="color:white; margin:0;">🛡️ RiskSentinel</h1>'
    f'<p style="color:{PALETTE["text_muted"]}; margin:0;">Agentic Systemic Risk Simulator — S&P 500 Financial Contagion</p>'
    '</div>',
    unsafe_allow_html=True,
)
tab_simulate, tab_dashboard, tab_explain, tab_settings = st.tabs(
    ["🌐 Simulate", "📊 Dashboard", "🔍 Explainability", "⚙️ Settings"]
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
        st.caption("Dettagli completi nel tab Dashboard.")

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
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = pool.submit(callable_fn)
            try:
                while not future.done():
                    _heartbeat(phase_text, eta_sec - (time.perf_counter() - t_start))
                    if st.session_state.run_cancel_requested:
                        future.cancel()
                        raise RuntimeError("RunCancelledByUser")
                    time.sleep(0.25)
                return future.result()
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

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
    cols = st.columns(7 if sr else 5)
    cols[0].metric("Nodes", m["n_nodes"])
    cols[1].metric("Edges", f"{m['n_edges']:,}")
    cols[2].metric("Density", f"{m['density']:.3f}")
    cols[3].metric("Regime", gd["regime"])
    cols[4].metric("VIX", f"{gd['vix']:.1f}")
    if sr:
        summary = sr.summary()
        cols[5].metric("Cascade", f"{summary['cascade_depth']} waves")
        cols[6].metric("Avg Stress", f"{summary['avg_stress']*100:.1f}%")


with tab_simulate:
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

with tab_dashboard:
    st.markdown("### 🛰️ Live Risk Dashboard")
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

with tab_explain:
    if not st.session_state.show_explainability:
        st.info("Enable `Show explainability panel` from `Settings` to view trace details.")
    elif st.session_state.run_trace:
        trace = st.session_state.run_trace
        st.markdown("### 🔍 Explainability")
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
            "🧾 Explainability (.json)",
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

with tab_settings:
    st.markdown("### 🤖 GPT Orchestrator Settings")
    is_agent_ready, agent_err = get_agent_config_status()
    access_policy = get_gpt_access_policy()
    can_enable_agent_mode = bool(is_agent_ready and access_policy["allowed"])
    if can_enable_agent_mode:
        st.caption("GPT orchestrator available")
    elif not is_agent_ready:
        st.caption("GPT orchestrator unavailable (using local fallback)")
    else:
        st.caption("GPT locked: judge access code required")

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

    if access_policy["gate_enabled"] and not access_policy["allowed"]:
        judge_code = st.text_input("Judge access code", type="password", placeholder="Enter code")
        if st.button("Unlock GPT", use_container_width=True):
            if unlock_judge_access(judge_code):
                st.session_state.judge_unlock_error = ""
                st.rerun()
            else:
                st.session_state.judge_unlock_error = "Invalid code."
        if st.session_state.judge_unlock_error:
            st.error(st.session_state.judge_unlock_error)
    elif access_policy["gate_enabled"] and access_policy["allowed"]:
        st.success("Judge access unlocked for this session.")

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
        f"{_get_runtime_int('GPT_MAX_CALLS_PER_SESSION', 120)} per session."
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
