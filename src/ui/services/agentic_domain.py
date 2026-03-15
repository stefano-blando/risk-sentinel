"""Agentic domain helpers and prompt/context utilities."""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone


def build_context_facts_html(graph_data: dict | None) -> str:
    """Fallback deterministic facts when no parsed shock scenario is present."""
    if not graph_data:
        return ""
    metrics = graph_data.get("metrics", {})
    lines = [
        "<b>Context Facts (Deterministic)</b>",
        (
            f"• Date: {html.escape(str(graph_data.get('date', 'n/a')))} | "
            f"Regime: {html.escape(str(graph_data.get('regime', 'n/a')))} "
            f"(VIX {graph_data.get('vix', 0.0):.1f})"
        ),
        (
            f"• Network: nodes {metrics.get('n_nodes', 'n/a')} | "
            f"edges {metrics.get('n_edges', 'n/a')} | "
            f"density {metrics.get('density', 0.0):.3f}"
        ),
    ]
    return "<br>".join(lines)


def build_memory_hint(query: str, history: list[dict], top_k: int = 2) -> str:
    """Retrieve brief episodic memory from prior runs using token overlap."""
    tokens = {
        word
        for word in re.findall(r"[a-zA-Z]{3,}", query.lower())
        if word not in {"what", "with", "that", "from"}
    }
    if not tokens:
        return ""
    scored: list[tuple[int, dict]] = []
    for row in history:
        prev_query = str(row.get("query", "")).lower()
        if not prev_query:
            continue
        prev_tokens = set(re.findall(r"[a-zA-Z]{3,}", prev_query))
        score = len(tokens & prev_tokens)
        if score <= 0:
            continue
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
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
    *,
    user_query: str,
    facts_plain: str,
    risk_profile: str,
    risk_profile_guidance: dict[str, str],
    memory_hint: str = "",
    rag_context: str = "",
    evidence_gate_strict: bool = True,
) -> str:
    profile_hint = risk_profile_guidance.get(risk_profile, risk_profile_guidance["balanced"])
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


def remember_session_decision(query: str, trace: dict, session_state) -> None:
    """Store concise session memory record for next runs."""
    result = trace.get("result", {}) if isinstance(trace, dict) else {}
    policy = trace.get("policy", {}) if isinstance(trace, dict) else {}
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": query[:220],
        "state": result.get("state", "n/a"),
        "risk_profile": policy.get("risk_profile", session_state.risk_profile),
        "route": (policy.get("router") or {}).get("route", "n/a")
        if isinstance(policy.get("router"), dict)
        else "n/a",
        "critic_approved": result.get("critic_approved"),
        "uncertainty_score": result.get("uncertainty_score"),
    }
    memory = list(session_state.session_decisions or [])
    memory.append(entry)
    session_state.session_decisions = memory[-40:]


def build_session_decision_hint(
    query: str,
    records: list[dict],
    tokenize_fn,
    similarity_fn,
    top_k: int = 2,
) -> str:
    """Semantic hint from recent decisions in this Streamlit session."""
    if not records:
        return ""
    query_tokens = tokenize_fn(query)
    scored: list[tuple[float, dict]] = []
    for row in records:
        prev_query = str(row.get("query", ""))
        prev_tokens = tokenize_fn(prev_query)
        score = similarity_fn(query_tokens, prev_tokens)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    lines = []
    for _, row in scored[:top_k]:
        lines.append(
            f"- memory_query='{row.get('query', '')[:80]}' | state={row.get('state')} | "
            f"risk_profile={row.get('risk_profile')} | critic_approved={row.get('critic_approved')}"
        )
    return "\n".join(lines)


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

    rate_limit_events = sum(1 for event in events if event.get("label") in {"gpt_backoff", "gpt_policy_block"})
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


def format_llm_text_for_card(text: str) -> str:
    """Render LLM markdown-like output as readable HTML for message cards."""
    output_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            output_lines.append("")
            continue
        line = line.replace("**", "").replace("`", "")
        if line.startswith("### "):
            output_lines.append(f"<b>{html.escape(line[4:])}</b>")
            continue
        if line.startswith("## "):
            output_lines.append(f"<b>{html.escape(line[3:])}</b>")
            continue
        if line.startswith("# "):
            output_lines.append(f"<b>{html.escape(line[2:])}</b>")
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and all(set(cell) <= set("-:") for cell in cells):
                continue
            row = " - ".join(cell for cell in cells if cell)
            output_lines.append(html.escape(row))
            continue
        if line.startswith("- ") or line.startswith("* "):
            output_lines.append(f"• {html.escape(line[2:])}")
            continue
        output_lines.append(html.escape(line))
    return "<br>".join(output_lines)


def build_agent_cache_key(
    *,
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
    cache: dict,
    exact_key: str,
    query: str,
    fingerprint: dict,
    tokenize_fn,
    similarity_fn,
    cache_semantic_min_score: float,
) -> tuple[dict | None, str]:
    exact = cache.get(exact_key)
    if exact:
        return exact, "exact"

    query_tokens = tokenize_fn(query)
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
        entry_tokens = set(entry.get("query_tokens", []))
        score = similarity_fn(query_tokens, entry_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= cache_semantic_min_score:
        return best_entry, f"semantic:{best_score:.2f}"
    return None, "miss"


def get_agent_config_status(get_settings_fn) -> tuple[bool, str]:
    """Validate env vars needed by Agent Framework chat client."""
    try:
        settings = get_settings_fn()
    except Exception as exc:
        return False, str(exc)

    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    missing = [key for key in required if not str(getattr(settings, key, "")).strip()]
    if missing:
        return False, f"Missing env vars: {', '.join(missing)}"
    endpoint = str(getattr(settings, "AZURE_OPENAI_ENDPOINT", "")).strip()
    if ".openai.azure.com" not in endpoint:
        return False, "AZURE_OPENAI_ENDPOINT must point to your Azure OpenAI resource (*.openai.azure.com)."
    try:
        from src.utils.azure_config import get_agent_framework_chat_client

        client = get_agent_framework_chat_client(
            deployment_name=str(getattr(settings, "AZURE_OPENAI_FALLBACK_DEPLOYMENT", "")).strip()
            or str(getattr(settings, "AZURE_OPENAI_DEPLOYMENT", "")).strip()
        )
        if not hasattr(client, "as_agent"):
            return False, "Installed agent-framework client is incompatible: missing as_agent(). Pin agent-framework==1.0.0rc1."
    except Exception as exc:
        return False, f"Agent runtime init failed: {type(exc).__name__}: {exc}"
    return True, ""
