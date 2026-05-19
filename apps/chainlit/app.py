"""RiskSentinel Chainlit app.

Deterministic local systemic-risk simulation chat with optional GPT control-plane analysis.

Run:
  ./venv/bin/python -m chainlit run apps/chainlit/app.py -w

Optional GPT mode:
  CHAINLIT_USE_GPT=1 ./venv/bin/python -m chainlit run apps/chainlit/app.py -w
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import chainlit as cl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import run_parallel_workflow
from src.core import contagion, data_loader, network
from src.utils.azure_config import get_agent_framework_chat_client, get_settings

DEFAULT_DATE = "2025-12-01"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL = "debtrank"
MAX_COMPARE_TICKERS = 8

COMPANY_NAME_MAP = {
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "GOLDMAN": "GS",
    "GOLDMAN SACHS": "GS",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "META": "META",
    "EXXON": "XOM",
    "CHEVRON": "CVX",
}


@lru_cache(maxsize=1)
def _get_static_data() -> tuple[dict[str, str], list[str]]:
    return data_loader.get_sector_dict(), data_loader.get_ticker_list()


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _extract_tickers(query: str, tickers: list[str]) -> list[str]:
    query_upper = query.upper().strip()
    found: list[str] = []

    for name, ticker in COMPANY_NAME_MAP.items():
        if name in query_upper and ticker not in found:
            found.append(ticker)

    for ticker in tickers:
        if re.search(rf"\b{re.escape(ticker)}\b", query_upper) and ticker not in found:
            found.append(ticker)

    return found


def _infer_model(query: str) -> str:
    q = query.lower()
    if "linear threshold" in q or "linear_threshold" in q:
        return "linear_threshold"
    if "cascade removal" in q or "cascade_removal" in q:
        return "cascade_removal"
    return DEFAULT_MODEL


def _parse_query(query: str, tickers: list[str]) -> dict[str, Any] | None:
    found_tickers = _extract_tickers(query, tickers)
    if not found_tickers:
        return None

    shock = 50
    pct_match = re.search(r"(\d{1,3})\s*%", query)
    if pct_match:
        shock = min(100, max(10, int(pct_match.group(1))))

    date = DEFAULT_DATE
    date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", query)
    if date_match:
        date = date_match.group(1)

    compare_intent = any(sig in query.lower() for sig in ["compare", "rank", "between", " vs ", "versus"])

    return {
        "ticker": found_tickers[0],
        "tickers": found_tickers,
        "shock": shock,
        "date": date,
        "model": _infer_model(query),
        "compare": compare_intent and len(found_tickers) >= 2,
    }


def _build_graph(date: str, threshold: float = DEFAULT_THRESHOLD):
    sector_dict, _ = _get_static_data()
    corr, actual_date = data_loader.get_correlation_matrix(date)
    graph = network.build_network(corr, threshold=threshold, sector_dict=sector_dict)

    regimes = data_loader.load_regime_data()
    ts = data_loader.find_nearest_date(date, regimes.index.tolist())
    regime_row = regimes.loc[ts]

    context = {
        "date": str(actual_date.date()),
        "regime": str(regime_row["Regime"]),
        "vix": float(regime_row["VIX"]),
        "threshold": threshold,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
    }
    return graph, context


def _format_single_response(parsed: dict[str, Any], context: dict[str, Any], result: contagion.ShockResult) -> str:
    sector_dict, _ = _get_static_data()
    s = result.summary()
    top_nodes = s.get("top_10_affected", [])[:5]

    lines = [
        "## Deterministic Simulation",
        (
            f"- **Date**: {context['date']} | **Regime**: {context['regime']} "
            f"(VIX {context['vix']:.1f}) | **Threshold**: {context['threshold']:.2f}"
        ),
        f"- **Scenario**: {parsed['ticker']} shock {parsed['shock']}% with `{parsed['model']}`",
        (
            f"- **Impact**: affected {s['n_affected']} / {context['nodes']} nodes, "
            f"defaulted {s['n_defaulted']}, cascade {s['cascade_depth']} waves"
        ),
        f"- **Stress**: total {s['total_stress']:.2f}, average {s['avg_stress']*100:.2f}%",
        "",
        "### Top Affected Nodes",
    ]

    if top_nodes:
        for item in top_nodes:
            ticker = item["ticker"]
            sector = sector_dict.get(ticker, "Unknown")
            lines.append(f"- {ticker} ({sector}): {item['stress']*100:.1f}%")
    else:
        lines.append("- No affected nodes detected above baseline.")

    return "\n".join(lines)


def _format_compare_response(
    parsed: dict[str, Any],
    context: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        "## Deterministic Compare",
        (
            f"- **Date**: {context['date']} | **Regime**: {context['regime']} "
            f"(VIX {context['vix']:.1f}) | **Shock**: {parsed['shock']}% | **Model**: `{parsed['model']}`"
        ),
        f"- **Tickers compared**: {', '.join([r['ticker'] for r in rows])}",
        "",
        "| Rank | Ticker | Affected | Defaulted | Waves | Total Stress | Avg Stress % |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]

    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['ticker']} | {row['n_affected']} | {row['n_defaulted']} | "
            f"{row['cascade_depth']} | {row['total_stress']:.2f} | {row['avg_stress_pct']:.2f} |"
        )

    lines.append("")
    lines.append("### Takeaway")
    if rows:
        top = rows[0]
        lines.append(
            f"- Highest systemic impact in this run: **{top['ticker']}** "
            f"(total stress {top['total_stress']:.2f}, {top['cascade_depth']} waves)."
        )

    return "\n".join(lines)


def _run_local_analysis(query: str) -> tuple[str, dict[str, Any] | None]:
    sector_dict, tickers = _get_static_data()
    parsed = _parse_query(query, tickers)
    if not parsed:
        help_text = (
            "Posso analizzare scenari tipo:\n"
            "- `What if JPM crashes 40% on 2025-12-01?`\n"
            "- `Compare JPM GS 50% on 2025-12-01`\n"
            "- `Simulate NVDA 60%`"
        )
        return help_text, None

    graph, context = _build_graph(parsed["date"], DEFAULT_THRESHOLD)

    if parsed["compare"]:
        rows: list[dict[str, Any]] = []
        for ticker in parsed["tickers"][:MAX_COMPARE_TICKERS]:
            if ticker not in graph:
                continue
            result = contagion.run_shock_scenario(graph, ticker, parsed["shock"] / 100.0, parsed["model"])
            s = result.summary()
            rows.append(
                {
                    "ticker": ticker,
                    "n_affected": s["n_affected"],
                    "n_defaulted": s["n_defaulted"],
                    "cascade_depth": s["cascade_depth"],
                    "total_stress": s["total_stress"],
                    "avg_stress_pct": s["avg_stress"] * 100,
                }
            )

        rows.sort(key=lambda x: (-x["total_stress"], -x["cascade_depth"], x["ticker"]))
        return _format_compare_response(parsed, context, rows), parsed

    ticker = parsed["ticker"]
    if ticker not in graph:
        return f"Ticker `{ticker}` non trovato nel grafo per `{parsed['date']}` con threshold {DEFAULT_THRESHOLD:.2f}.", parsed

    result = contagion.run_shock_scenario(graph, ticker, parsed["shock"] / 100.0, parsed["model"])
    return _format_single_response(parsed, context, result), parsed


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    obj = re.search(r"(\{[\s\S]*\})", text)
    if obj:
        candidates.append(obj.group(1).strip())

    for raw in candidates:
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _format_gpt_payload(text: str) -> str:
    payload = _extract_json_payload(text)
    if not payload:
        return f"## GPT Control-Plane\n\n{text}"

    def _list(title: str, key: str) -> list[str]:
        val = payload.get(key, [])
        if isinstance(val, list) and val:
            out = [f"### {title}"]
            out.extend([f"- {str(x)}" for x in val])
            return out
        return []

    lines = ["## GPT Control-Plane"]
    lines.extend(_list("Situation", "situation"))
    lines.extend(_list("Quant Results", "quant_results"))
    if payload.get("risk_rating"):
        lines.append(f"### Risk Rating\n- **{payload.get('risk_rating')}**")
    lines.extend(_list("Actions", "actions"))
    lines.extend(_list("Monitoring", "monitoring_triggers"))
    lines.extend(_list("Evidence Used", "evidence_used"))

    if payload.get("validation"):
        val = payload["validation"]
        if isinstance(val, dict):
            approved = val.get("critic_approved")
            if approved is not None:
                lines.append(f"### Critic\n- approved: **{bool(approved)}**")

    if payload.get("uncertainty_score") is not None:
        lines.append(f"- uncertainty: `{payload.get('uncertainty_score')}`")

    if payload.get("notes"):
        lines.append(f"### Notes\n- {payload.get('notes')}")

    return "\n\n".join(lines)


def _can_use_gpt() -> tuple[bool, str]:
    if os.getenv("CHAINLIT_USE_GPT", "0").strip() != "1":
        return False, "CHAINLIT_USE_GPT!=1"

    try:
        settings = get_settings()
    except Exception as exc:
        return False, f"settings_error:{type(exc).__name__}"

    required = [
        str(getattr(settings, "AZURE_OPENAI_ENDPOINT", "")).strip(),
        str(getattr(settings, "AZURE_OPENAI_API_KEY", "")).strip(),
        str(getattr(settings, "AZURE_OPENAI_DEPLOYMENT", "")).strip(),
    ]
    if not all(required):
        return False, "azure_env_missing"

    return True, "ok"


def _should_use_gpt(query: str, parsed: dict[str, Any] | None) -> bool:
    if not parsed:
        return True
    q = query.lower()
    if parsed.get("compare"):
        return True
    return any(sig in q for sig in ["portfolio", "hedg", "strategy", "recommend", "advice"])


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "# RiskSentinel (Chainlit)\n"
            "Scrivi uno scenario, ad esempio:\n"
            "- `What if JPM crashes 40% on 2025-12-01?`\n"
            "- `Compare JPM GS 50% on 2025-12-01`\n"
            "\n"
            "Default: simulazione locale deterministica.\n"
            "Per attivare GPT control-plane: `CHAINLIT_USE_GPT=1`"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = _normalize_query(message.content)
    progress = cl.Message(content="Analizzo scenario...")
    await progress.send()

    local_text, parsed = await asyncio.to_thread(_run_local_analysis, query)

    gpt_enabled, gpt_reason = _can_use_gpt()
    gpt_text = ""
    if gpt_enabled and _should_use_gpt(query, parsed):
        try:
            client = get_agent_framework_chat_client()
            raw = await asyncio.wait_for(
                run_parallel_workflow(client, query, timeout_sec=45),
                timeout=50,
            )
            gpt_text = _format_gpt_payload(raw)
        except Exception as exc:
            gpt_text = f"## GPT Control-Plane\n\n⚠️ errore: `{type(exc).__name__}: {exc}`"
    elif os.getenv("CHAINLIT_USE_GPT", "0").strip() == "1":
        gpt_text = f"## GPT Control-Plane\n\nSkipped (`{gpt_reason}` or query not requiring GPT)."

    body = local_text if not gpt_text else f"{local_text}\n\n---\n\n{gpt_text}"
    await progress.update(content=body)
