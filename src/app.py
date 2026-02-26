"""
RiskSentinel — Streamlit App
Agentic Systemic Risk Simulator for Financial Contagion
"""

import asyncio
import sys
import html
import time
import json
import os
import hmac
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

from src.agents.orchestrator import create_orchestrator, create_simple_agent, run_query
from src.core import data_loader, network, contagion
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
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .agent-msg {
        background: #1a1f2e;
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
    }
    .agent-architect { border-color: #4fc3f7; }
    .agent-quant { border-color: #ff8a65; }
    .agent-advisor { border-color: #81c784; }
    .agent-sentinel { border-color: #ce93d8; }
    .risk-low { color: #00d26a; }
    .risk-moderate { color: #ffc107; }
    .risk-elevated { color: #ff8c00; }
    .risk-high { color: #ff4444; }
    .risk-critical { color: #ff0000; font-weight: bold; }
    .main-header {
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .sector-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
    }
    .crisis-btn button {
        font-size: 12px !important;
    }
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
}

BENCHMARK_QUERIES = [
    "What happens if JPM crashes 40% on 2025-12-01?",
    "Simulate AAPL 60% shock on 2024-08-05 with debtrank.",
    "Compare systemic risk between JPM and GS on 2025-12-01.",
    "Run linear threshold for NVDA 50% on 2025-12-01.",
    "Compare NVDA, XOM, and UNH with 50% shock on 2025-12-01.",
]

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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    }


def trace_event(trace: dict, label: str, detail: str = "") -> None:
    trace["events"].append(
        {
            "t_sec": round(time.perf_counter() - trace["_t0"], 3),
            "label": label,
            "detail": detail,
        }
    )


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
) -> list[dict]:
    """Compute deterministic metrics for multi-ticker comparison on same graph."""
    sector_dict = st.session_state.sector_dict
    rows: list[dict] = []
    for ticker in tickers[:3]:
        if ticker not in G:
            continue
        result = contagion.run_shock_scenario(G, ticker, shock_pct / 100.0, model)
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
    return rows


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


def build_agent_cache_key(
    query: str,
    strategy: str,
    primary_deployment: str,
    parsed: dict | None,
    threshold: float,
    model: str,
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
    }
    return json.dumps(payload, sort_keys=True)


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


def run_agent_query(
    query: str,
    timeout_sec: int = 35,
    strategy: str = "simple",
    deployment_name: str | None = None,
) -> str:
    if strategy == "orchestrator":
        return _run_async(_run_orchestrator_query_async(query, timeout_sec, deployment_name=deployment_name))
    return _run_async(_run_simple_query_async(query, timeout_sec, deployment_name=deployment_name))


def is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def run_agent_query_with_backoff(
    query: str,
    timeout_sec: int,
    strategy: str,
    deployment_name: str | None = None,
    max_retries: int = 2,
    base_delay_sec: float = 1.5,
    on_backoff: Callable[[float, int, int], None] | None = None,
) -> str:
    """Retry agent query on 429 with exponential backoff."""
    attempt = 0
    while True:
        try:
            return run_agent_query(
                query=query,
                timeout_sec=timeout_sec,
                strategy=strategy,
                deployment_name=deployment_name,
            )
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            wait_sec = base_delay_sec * (2 ** attempt)
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
    if avg_stress > 30:
        risk_level, risk_class = "CRITICAL", "risk-critical"
        advice = (f"Systemic event. Avg stress {avg_stress:.1f}%. "
                  f"<b>Act now:</b> (1) Broad hedges (SPY puts), "
                  f"(2) Liquidate high-centrality names, (3) Cash up.")
    elif avg_stress > 15:
        risk_level, risk_class = "HIGH", "risk-high"
        advice = (f"Severe contagion. Avg stress {avg_stress:.1f}%. "
                  f"<b>Actions:</b> (1) Sector hedges, "
                  f"(2) Review {ticker} counterparty exposure, (3) Tighten stops.")
    elif avg_stress > 5:
        risk_level, risk_class = "ELEVATED", "risk-elevated"
        advice = (f"Moderate contagion. Avg stress {avg_stress:.1f}%. "
                  f"<b>Monitor:</b> (1) VIX trajectory, "
                  f"(2) Direct {ticker} exposure, (3) No broad hedging yet.")
    else:
        risk_level, risk_class = "LOW", "risk-low"
        advice = f"Contained. Avg stress {avg_stress:.1f}%. Minimal systemic impact."

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
        return "#ff1744"
    elif stress >= 0.5:
        return "#ff6d00"
    elif stress >= 0.2:
        return "#ffc107"
    elif stress > 0.01:
        return "#4fc3f7"
    return "#2a3040"


def _build_base_layout(height: int = 580, title_text: str = "") -> go.Layout:
    """Shared Plotly layout for network graphs."""
    return go.Layout(
        showlegend=False, hovermode="closest",
        margin=dict(b=40, l=10, r=10, t=50),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        title=dict(text=title_text, font=dict(color="#ff8a65", size=16),
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
        line=dict(width=0.3, color="rgba(42, 48, 64, 0.25)"),
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
        line=dict(width=1.8, color="rgba(255, 100, 0, 0.45)"),
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
            colors.append("#ffffff" if is_s else _stress_color(s))
            sizes.append(28 if is_s else max(5, int(5 + s * 20)))
        else:
            colors.append(data_loader.SECTOR_COLORS.get(sector, "#cccccc"))
            sizes.append(7)

        s_txt = f"<br>Stress: {s:.1%}" if stress else ""
        texts.append(f"<b>{node}</b><br>Sector: {sector}<br>Connections: {G.degree(node)}{s_txt}")
        labels.append(node if (is_s or (stress and s >= 0.5)) else "")
        outlines.append("#ff0000" if is_s else "#0e1117")

    return go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        hoverinfo="text", hovertext=texts,
        marker=dict(size=sizes, color=colors, line=dict(width=1.5, color=outlines)),
        text=labels, textposition="top center",
        textfont=dict(size=9, color="white"),
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
            layout=go.Layout(title=dict(text=label, font=dict(color="#ff8a65", size=16),
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
                              font=dict(color="#ff8a65", size=13)),
            tickcolor="#8b95a5", font=dict(color="#8b95a5"),
            bgcolor="#1a1f2e", bordercolor="#1a1f2e",
            activebgcolor="#ff8a65",
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


def build_timeline_figure() -> go.Figure | None:
    """Mini timeline chart: network density + VIX over time."""
    try:
        nm = data_loader.load_network_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nm.index, y=nm["density"], name="Density",
            line=dict(color="#4fc3f7", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=nm.index, y=nm["vix"] / 100, name="VIX / 100",
            line=dict(color="#ff8a65", width=1.5),
            yaxis="y2",
        ))

        # Mark crises
        for name, (start, end) in data_loader.CRISIS_EVENTS.items():
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(255, 0, 0, 0.1)", line_width=0,
                annotation_text=name.split(" ")[0],
                annotation_position="top left",
                annotation_font_size=8,
                annotation_font_color="#ff8a65",
            )

        # Mark selected date
        if st.session_state.graph_data:
            fig.add_vline(
                x=st.session_state.graph_data["date"],
                line_dash="dash", line_color="white", line_width=1,
            )

        fig.update_layout(
            height=200, margin=dict(l=40, r=40, t=20, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1, font=dict(size=10, color="#8b95a5")),
            xaxis=dict(showgrid=False, color="#8b95a5"),
            yaxis=dict(title="Density", showgrid=True, gridcolor="#1a1f2e", color="#4fc3f7", range=[0, 0.8]),
            yaxis2=dict(title="VIX/100", overlaying="y", side="right", showgrid=False,
                        color="#ff8a65", range=[0, 0.8]),
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


def generate_trace_bundle_json() -> str:
    """Downloadable explainability payload for audit/demo."""
    payload = {
        "last_run_metrics": st.session_state.last_run_metrics,
        "trace": st.session_state.run_trace,
        "history_size": len(st.session_state.run_trace_history),
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ RiskSentinel")
    st.caption("Agentic Systemic Risk Simulator")
    st.divider()

    # === AGENT MODE ===
    st.markdown("### 🤖 Agent Mode")
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
        options=["simple", "orchestrator"],
        index=0 if st.session_state.agent_strategy == "simple" else 1,
        disabled=not st.session_state.agent_mode,
        help="simple = one tool-calling agent (faster). orchestrator = multi-agent chain (richer but slower).",
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
    st.session_state.agent_timeout_sec = st.slider(
        "Agent timeout (sec)",
        min_value=10, max_value=120, value=st.session_state.agent_timeout_sec, step=5,
        disabled=not st.session_state.agent_mode,
    )
    if st.button("Run GPT Diagnostic", use_container_width=True, disabled=not can_enable_agent_mode):
        try:
            st.session_state.agent_diagnostic = run_gpt_diagnostic()
        except Exception as exc:
            st.session_state.agent_diagnostic = f"diagnostic_err={type(exc).__name__}: {exc}"
    try:
        primary_dep, fallback_dep = get_deployment_routing(st.session_state.high_quality_mode)
        st.caption(f"Primary: {primary_dep} | Fallback: {fallback_dep}")
    except Exception:
        pass
    if not is_agent_ready and agent_err:
        st.warning(agent_err)
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

    rate_cfg_caption = (
        f"Soft limits: {_get_runtime_int('GPT_MAX_CALLS_PER_MINUTE_SESSION', 8)}/min session, "
        f"{_get_runtime_int('GPT_MAX_CALLS_PER_MINUTE_GLOBAL', 20)}/min global, "
        f"{_get_runtime_int('GPT_MAX_CALLS_PER_SESSION', 120)} per session."
    )
    st.caption(rate_cfg_caption)
    if st.session_state.agent_diagnostic:
        st.code(st.session_state.agent_diagnostic, language="text")

    st.divider()

    # === DEMO MODE ===
    st.markdown("### 🎬 Demo Mode")
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

    st.divider()

    # === OBSERVABILITY ===
    st.markdown("### 🔍 Explainability")
    st.session_state.show_explainability = st.toggle(
        "Show explainability panel",
        value=st.session_state.show_explainability,
    )
    st.session_state.persist_trace_logs = st.toggle(
        "Persist traces to JSONL",
        value=st.session_state.persist_trace_logs,
        help="Stores each run trace in artifacts/run_traces.jsonl",
    )

    if st.button("🧪 Run Local Benchmark (5 queries)", use_container_width=True):
        with st.spinner("Running benchmark pack..."):
            st.session_state.eval_results = run_local_benchmark(st.session_state.sel_threshold)
    if st.session_state.eval_results:
        er = st.session_state.eval_results
        bcols = st.columns(3)
        bcols[0].metric("Benchmark", f"{er['n_ok']}/{er['n_queries']}")
        bcols[1].metric("Success", f"{er['success_rate_pct']:.1f}%")
        bcols[2].metric("Avg latency", f"{er['avg_latency_s']:.2f}s")

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

    st.divider()
    st.markdown("### 🏷️ Sectors")
    for sector, color in data_loader.SECTOR_COLORS.items():
        short = sector.replace("Information ", "Info ").replace("Consumer ", "Cons. ").replace("Communication ", "Comm. ")
        st.markdown(f'<span class="sector-dot" style="background:{color}"></span><small>{short}</small>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# BUILD / SHOCK ACTIONS
# ---------------------------------------------------------------------------
if build_btn or st.session_state.graph_data is None:
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


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    '<h1 style="color:white; margin:0;">🛡️ RiskSentinel</h1>'
    '<p style="color:#90caf9; margin:0;">Agentic Systemic Risk Simulator — S&P 500 Financial Contagion</p>'
    '</div>',
    unsafe_allow_html=True,
)

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
    complex_query = is_complex_query(chat_query)
    in_scope, scope_reason = is_query_in_scope(chat_query, parsed)
    model_for_query = infer_model_from_query(chat_query)
    runtime_access_policy = get_gpt_access_policy()
    primary_deployment, fallback_deployment = get_deployment_routing(st.session_state.high_quality_mode)
    st.session_state.agent_messages = []
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
        "primary_deployment": primary_deployment,
        "fallback_deployment": fallback_deployment,
        "gpt_access_allowed": runtime_access_policy["allowed"],
        "gpt_access_reason": runtime_access_policy["reason"],
    }
    trace_event(trace, "query_received", chat_query)
    trace_event(trace, "scope_check", scope_reason)

    with st.status("Processing query...", expanded=True) as status:
        progress = st.progress(5)
        phase = st.empty()
        elapsed_box = st.empty()

        def _step(pct: int, text: str) -> None:
            phase.markdown(f"**{text}**")
            progress.progress(pct)
            elapsed_box.caption(f"Elapsed: {time.perf_counter() - t_start:.1f}s")

        _step(10, "Parsing user input")
        trace_event(trace, "parse_complete", f"parsed={bool(parsed)}, complex={complex_query}")

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

        else:
            # Local deterministic path (used both for fast answers and deterministic facts anchoring).
            run_local_first = bool(parsed)
            suppress_local_messages = bool(complex_query and st.session_state.agent_mode)
            if run_local_first:
                _step(30, "Running local network build and contagion simulation")
                trace_event(trace, "local_start", f"ticker={parsed['ticker']}, shock={parsed['shock']}")
                t_local = time.perf_counter()
                date = parsed.get("date") or selected_date
                if not suppress_local_messages:
                    st.session_state.agent_messages.append(
                        ("Sentinel", "🛡️", "agent-sentinel",
                         f'Understanding query: "<i>{chat_query}</i>"<br>'
                         f'→ Target: <b>{parsed["ticker"]}</b>, Shock: <b>{parsed["shock"]}%</b>')
                    )
                G = do_build_network(date, threshold, emit_messages=not suppress_local_messages)
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

            should_run_gpt = (
                st.session_state.agent_mode
                and in_scope
                and runtime_access_policy["allowed"]
                and (not parsed or st.session_state.gpt_for_parseable_queries or complex_query)
            )
            gpt_policy_block_reason = ""
            if not runtime_access_policy["allowed"]:
                gpt_policy_block_reason = f"access_locked:{runtime_access_policy['reason']}"
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
                gpt_attempted = True
                t_gpt = time.perf_counter()
                strategy = st.session_state.agent_strategy
                cache_key = build_agent_cache_key(
                    query=chat_query,
                    strategy=strategy,
                    primary_deployment=primary_deployment,
                    parsed=parsed,
                    threshold=threshold,
                    model=model_for_query,
                )
                facts_html = build_simulation_facts_html()
                facts_mode = "single"
                if parsed and complex_query and len(parsed.get("tickers", [])) >= 2 and st.session_state.graph_data:
                    gd = st.session_state.graph_data
                    compare_rows = _compute_compare_rows(
                        G=gd["G"],
                        tickers=parsed["tickers"],
                        shock_pct=parsed["shock"],
                        model=model_for_query,
                    )
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

                if facts_html:
                    facts_plain = re.sub(r"<[^>]+>", "", facts_html).replace("&nbsp;", " ")
                    trace["policy"]["facts_preview"] = facts_plain[:500]
                    prompt_for_agent = (
                        "Use the deterministic simulation facts below exactly as numeric ground truth. "
                        "Do not invent or override these values.\n\n"
                        f"{facts_plain}\n\n"
                        f"User request:\n{chat_query}"
                    )
                else:
                    prompt_for_agent = chat_query

                def _push_gpt_message(answer_text: str, deployment_used: str, label_strategy: str, cached: bool = False):
                    formatted_answer = format_llm_text_for_card(answer_text)
                    body = f"{facts_html}<br><br>{formatted_answer}" if facts_html else formatted_answer
                    suffix = ", cached" if cached else ""
                    st.session_state.agent_messages.append(
                        ("Sentinel", "🛡️", "agent-sentinel",
                         f"<b>Agent ({label_strategy}, {deployment_used}{suffix})</b><br>{body}")
                    )

                def _on_backoff(wait_sec: float, retry_idx: int, max_attempts: int) -> None:
                    _step(
                        70,
                        f"Rate limited (429). Waiting {wait_sec:.1f}s before retry {retry_idx + 1}/{max_attempts}",
                    )
                    trace_event(trace, "gpt_backoff", f"wait={wait_sec:.1f}s retry={retry_idx + 1}/{max_attempts}")

                cached = st.session_state.agent_response_cache.get(cache_key)
                trace["policy"]["cache_hit"] = bool(cached)
                if cached:
                    _step(60, "Using cached GPT analysis")
                    _push_gpt_message(
                        answer_text=cached["answer"],
                        deployment_used=cached.get("deployment", primary_deployment),
                        label_strategy=cached.get("strategy", strategy),
                        cached=True,
                    )
                    gpt_success = True
                    engine_label = f"{cached.get('deployment', primary_deployment)} ({strategy}, cached)"
                    run_state = "gpt_cached"
                    trace_event(trace, "gpt_cache_hit", engine_label)
                else:
                    try:
                        register_gpt_call()
                        _step(60, f"Running GPT analysis ({strategy})")
                        answer = run_agent_query_with_backoff(
                            prompt_for_agent,
                            timeout_sec=st.session_state.agent_timeout_sec,
                            strategy=strategy,
                            deployment_name=primary_deployment,
                            max_retries=2,
                            on_backoff=_on_backoff,
                        )
                        _push_gpt_message(answer, primary_deployment, strategy)
                        st.session_state.agent_response_cache[cache_key] = {
                            "answer": answer,
                            "deployment": primary_deployment,
                            "strategy": strategy,
                            "ts": time.time(),
                        }
                        gpt_success = True
                        engine_label = f"{primary_deployment} ({strategy})"
                        run_state = "gpt_ok"
                        trace_event(trace, "gpt_ok", engine_label)
                    except Exception as exc:
                        trace_event(trace, "gpt_err", f"{type(exc).__name__}: {str(exc)[:120]}")
                        # If full orchestrator times out, retry once with simple agent.
                        retried = False
                        if st.session_state.agent_strategy == "orchestrator":
                            try:
                                _step(72, "Orchestrator timeout. Retrying with simple strategy")
                                answer = run_agent_query_with_backoff(
                                    prompt_for_agent,
                                    timeout_sec=min(45, st.session_state.agent_timeout_sec),
                                    strategy="simple",
                                    deployment_name=primary_deployment,
                                    max_retries=1,
                                    on_backoff=_on_backoff,
                                )
                                _push_gpt_message(answer, primary_deployment, "simple")
                                st.session_state.agent_response_cache[cache_key] = {
                                    "answer": answer,
                                    "deployment": primary_deployment,
                                    "strategy": "simple",
                                    "ts": time.time(),
                                }
                                retried = True
                                gpt_success = True
                                engine_label = f"{primary_deployment} (simple retry)"
                                run_state = "gpt_retry_ok"
                                trace_event(trace, "gpt_retry_ok", engine_label)
                            except Exception:
                                retried = False

                        # If we hit Azure TPM/RPM limits, retry on fallback deployment.
                        if not retried and is_rate_limit_error(exc):
                            try:
                                _step(80, f"Rate limit on {primary_deployment}. Retrying with {fallback_deployment}")
                                answer = run_agent_query_with_backoff(
                                    prompt_for_agent,
                                    timeout_sec=min(45, st.session_state.agent_timeout_sec),
                                    strategy="simple",
                                    deployment_name=fallback_deployment,
                                    max_retries=1,
                                    on_backoff=_on_backoff,
                                )
                                _push_gpt_message(answer, fallback_deployment, "simple")
                                st.session_state.agent_response_cache[cache_key] = {
                                    "answer": answer,
                                    "deployment": fallback_deployment,
                                    "strategy": "simple",
                                    "ts": time.time(),
                                }
                                retried = True
                                gpt_success = True
                                engine_label = f"{fallback_deployment} (fallback)"
                                run_state = "gpt_fallback_ok"
                                trace_event(trace, "gpt_fallback_ok", engine_label)
                            except Exception:
                                retried = False

                        if not retried:
                            st.session_state.agent_messages.append(
                                ("Sentinel", "🛡️", "agent-sentinel",
                                 f"⚠️ GPT agent failed ({type(exc).__name__}: {html.escape(str(exc))[:220]}). "
                                 f"Using local simulation output only.")
                            )
                            if parsed and local_sec is None:
                                _step(88, "GPT unavailable. Running local fallback simulation")
                                t_local = time.perf_counter()
                                date = parsed.get("date") or selected_date
                                st.session_state.agent_messages.append(
                                    ("Sentinel", "🛡️", "agent-sentinel",
                                     f'Fallback local run<br>→ Target: <b>{parsed["ticker"]}</b>, Shock: <b>{parsed["shock"]}%</b>')
                                )
                                G = do_build_network(date, threshold)
                                do_run_shock(G, parsed["ticker"], parsed["shock"], model_for_query)
                                local_sec = time.perf_counter() - t_local
                                engine_label = "Local engine (after GPT failure)"
                                run_state = "gpt_failed_local_fallback"
                                trace_event(trace, "local_fallback_done", f"{local_sec:.3f}s")
                            elif not parsed:
                                st.session_state.agent_messages.append(
                                    ("Sentinel", "🛡️", "agent-sentinel",
                                     f'Could not parse query locally: "<i>{chat_query}</i>". '
                                     f'Try: "What if JPM crashes 40%?" or "Simulate AAPL 60% shock"')
                                )
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

    st.session_state.last_run_metrics = {
        "total_sec": elapsed,
        "local_sec": local_sec,
        "gpt_sec": gpt_sec,
        "gpt_attempted": gpt_attempted,
        "gpt_success": gpt_success,
        "engine": engine_label,
        "state": run_state,
        "gpt_calls_total_session": st.session_state.gpt_calls_total_session,
        "gpt_rate_limit_hits": st.session_state.gpt_rate_limit_hits,
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
    }
    trace_event(trace, "run_complete", run_state)
    trace = finalize_run_trace(trace)
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

# --- Main area ---
graph_col, info_col = st.columns([3, 2])

with graph_col:
    st.markdown("### 🌐 Correlation Network")

    if st.session_state.graph_data:
        G = st.session_state.graph_data["G"]
        pos = st.session_state.pos
        sr = st.session_state.shock_result

        if sr:
            blast_view = st.toggle("🎯 Blast radius only", value=False,
                                   help="Show only affected subgraph")
            fig = build_animated_figure(G, pos, sr, blast_radius_only=blast_view)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": True})
            st.markdown(
                "⚪ Shocked &nbsp; 🔴 Critical &nbsp; 🟠 High &nbsp; "
                "🟡 Moderate &nbsp; 🔵 Low &nbsp; ⚫ Unaffected &emsp; | &emsp; "
                "Use **▶ Play** or drag the **slider** below the graph")
        else:
            fig = build_graph_figure(G, pos)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": True})

    # Model comparison
    if st.session_state.comparison:
        st.markdown("### ⚖️ Model Comparison")
        comp = st.session_state.comparison
        comp_rows = []
        for model_name, res in comp.items():
            s = res.summary()
            comp_rows.append({
                "Model": model_name.replace("_", " ").title(),
                "Affected": s["n_affected"],
                "Defaulted": s["n_defaulted"],
                "Waves": s["cascade_depth"],
                "Avg Stress %": round(s["avg_stress"] * 100, 1),
                "Total Stress": round(s["total_stress"], 1),
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True,
                      column_config={
                          "Avg Stress %": st.column_config.ProgressColumn(
                              min_value=0, max_value=100, format="%.1f%%"),
                      })

        # Bar chart comparison
        fig_comp = go.Figure()
        model_colors = {"Linear Threshold": "#4fc3f7", "Debtrank": "#ff8a65", "Cascade Removal": "#ce93d8"}
        for _, row in comp_df.iterrows():
            fig_comp.add_trace(go.Bar(
                name=row["Model"],
                x=["Affected", "Defaulted", "Waves"],
                y=[row["Affected"], row["Defaulted"], row["Waves"]],
                marker_color=model_colors.get(row["Model"], "#888"),
                text=[row["Affected"], row["Defaulted"], row["Waves"]],
                textposition="auto",
            ))
        fig_comp.update_layout(
            barmode="group", height=250,
            margin=dict(l=40, r=20, t=20, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#8b95a5", size=11)),
            xaxis=dict(color="#8b95a5"), yaxis=dict(color="#8b95a5", showgrid=True, gridcolor="#1a1f2e"),
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

    # Network timeline
    st.markdown("### 📈 Network Health Timeline")
    timeline_fig = build_timeline_figure()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})


with info_col:
    st.markdown("### 🤖 Agent Analysis")
    run_metrics = st.session_state.last_run_metrics
    if run_metrics:
        badge_map = {
            "gpt_ok": "GPT OK",
            "gpt_cached": "GPT Cached",
            "gpt_retry_ok": "GPT Retry OK",
            "gpt_fallback_ok": "GPT Fallback OK",
            "gpt_failed": "GPT Failed (Local Fallback)",
            "gpt_failed_local_fallback": "GPT Failed (Local Fallback)",
            "local_fast_mode": "Local Fast Mode",
            "local_only": "Local Only",
            "parse_failed": "Parse Failed",
            "out_of_scope": "Out Of Scope",
            "gpt_policy_block_local": "GPT Blocked (Local Only)",
            "gpt_policy_block_parse_failed": "GPT Blocked + Parse Failed",
        }
        st.caption(f"Last run: {badge_map.get(run_metrics.get('state', ''), run_metrics.get('state', 'n/a'))}")
        tcols = st.columns(4)
        tcols[0].metric("Total", f"{run_metrics.get('total_sec', 0.0):.1f}s")
        local_val = run_metrics.get("local_sec")
        tcols[1].metric("Local", f"{local_val:.1f}s" if isinstance(local_val, float) else "n/a")
        gpt_val = run_metrics.get("gpt_sec")
        tcols[2].metric("GPT", f"{gpt_val:.1f}s" if isinstance(gpt_val, float) else "n/a")
        tcols[3].metric("Engine", run_metrics.get("engine", "n/a"))

    if st.session_state.agent_messages:
        for name, icon, css, text in st.session_state.agent_messages:
            agent_message(name, icon, css, text)
    else:
        st.info("Type a question below, use a **Crisis Preset**, or click **Build** → **Shock**.")

    if st.session_state.show_explainability and st.session_state.run_trace:
        trace = st.session_state.run_trace
        st.markdown("### 🔍 Explainability")
        p = trace.get("policy", {})
        r = trace.get("result", {})
        t = trace.get("timings", {})

        xcols = st.columns(3)
        xcols[0].metric("Route", r.get("state", "n/a"))
        xcols[1].metric("Cache hit", "Yes" if p.get("cache_hit") else "No")
        xcols[2].metric("In scope", "Yes" if trace.get("in_scope") else "No")

        with st.expander("Decision policy", expanded=False):
            st.json({
                "scope_reason": trace.get("scope_reason"),
                "complex_query": trace.get("complex_query"),
                "strategy": p.get("strategy"),
                "should_run_gpt": p.get("should_run_gpt"),
                "gpt_access_allowed": p.get("gpt_access_allowed"),
                "gpt_access_reason": p.get("gpt_access_reason"),
                "gpt_block_reason": p.get("gpt_block_reason", "none"),
                "facts_mode": p.get("facts_mode", "none"),
                "engine": r.get("engine"),
                "timings": t,
            })

        with st.expander("Execution trace", expanded=False):
            events = trace.get("events", [])
            if events:
                st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
            else:
                st.caption("No trace events.")

        if p.get("facts_preview"):
            with st.expander("Deterministic facts injected into GPT", expanded=False):
                st.code(p["facts_preview"], language="text")

        history = st.session_state.run_trace_history
        if history:
            states = [h.get("result", {}).get("state", "n/a") for h in history]
            hcols = st.columns(3)
            hcols[0].metric("Runs tracked", len(history))
            hcols[1].metric("GPT success", f"{sum(1 for h in history if h.get('result', {}).get('gpt_success'))}/{len(history)}")
            hcols[2].metric("Avg total", f"{np.mean([h.get('timings', {}).get('total_sec', 0.0) for h in history]):.1f}s")
            st.caption("Recent route states: " + ", ".join(pd.Series(states).value_counts().head(4).index.tolist()))
            st.caption(f"Session GPT calls: {st.session_state.gpt_calls_total_session} | Policy throttles: {st.session_state.gpt_rate_limit_hits}")

        if st.session_state.eval_results:
            st.markdown("### 🧪 Benchmark Results")
            er = st.session_state.eval_results
            st.caption(
                f"{er['n_ok']}/{er['n_queries']} success | "
                f"avg latency {er['avg_latency_s']:.2f}s | total {er['total_time_s']:.2f}s"
            )
            st.dataframe(pd.DataFrame(er["rows"]), use_container_width=True, hide_index=True)

    if st.session_state.shock_result:
        sr = st.session_state.shock_result

        # Sector impact
        st.markdown("### 📊 Sector Impact")
        sev_df = build_severity_df(sr)
        st.dataframe(sev_df, use_container_width=True, hide_index=True,
                      column_config={
                          "Avg Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                      })

        # Top nodes
        st.markdown("### 🎯 Most Vulnerable")
        affected = sr.affected_nodes[:10]
        if affected:
            df = pd.DataFrame(affected, columns=["Ticker", "Stress"])
            df["Sector"] = df["Ticker"].map(st.session_state.sector_dict)
            df["Stress %"] = (df["Stress"] * 100).round(1)
            st.dataframe(df[["Ticker", "Sector", "Stress %"]], use_container_width=True, hide_index=True,
                          column_config={
                              "Stress %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                          })

        # Download report
        st.divider()
        report = generate_report_text()
        brief_md = generate_report_markdown()
        trace_json = generate_trace_bundle_json()
        dcols = st.columns(3)
        dcols[0].download_button(
            "📥 Report (.txt)", report,
            file_name=f"risksentinel_report_{sr.shocked_node}_{st.session_state.graph_data['date']}.txt",
            mime="text/plain", use_container_width=True,
        )
        dcols[1].download_button(
            "📄 Executive Brief (.md)", brief_md,
            file_name=f"risksentinel_brief_{sr.shocked_node}_{st.session_state.graph_data['date']}.md",
            mime="text/markdown", use_container_width=True,
        )
        dcols[2].download_button(
            "🧾 Explainability (.json)", trace_json,
            file_name=f"risksentinel_trace_{sr.shocked_node}_{st.session_state.graph_data['date']}.json",
            mime="application/json", use_container_width=True,
        )


# Footer
st.divider()
st.caption(
    "RiskSentinel — Microsoft AI Dev Days Hackathon 2026 | "
    "Built with Microsoft Agent Framework, Azure AI Foundry, NetworkX, Streamlit"
)
