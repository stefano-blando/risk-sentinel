"""Pure query parsing and formatting helpers."""

from __future__ import annotations

import html
import json
import re


def extract_tickers_from_query(
    query: str,
    tickers: list[str],
    company_name_map: dict[str, str],
) -> list[str]:
    """Extract one or more tickers from user query using aliases + direct symbols."""
    query_upper = query.upper().strip()
    found: list[str] = []

    for name, ticker in company_name_map.items():
        if name in query_upper and ticker not in found:
            found.append(ticker)

    for ticker in tickers:
        if re.search(rf"\b{re.escape(ticker)}\b", query_upper) and ticker not in found:
            found.append(ticker)

    return found


def parse_chat_query(
    query: str,
    tickers: list[str],
    company_name_map: dict[str, str],
) -> dict | None:
    """Parse natural language shock queries into ticker/shock/date parameters."""
    found_tickers = extract_tickers_from_query(query, tickers, company_name_map)
    if not found_tickers:
        return None

    shock = 50
    pct_match = re.search(r"(\d+)\s*%", query)
    if pct_match:
        shock = min(100, max(10, int(pct_match.group(1))))

    date = None
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
    if date_match:
        date = date_match.group(1)

    return {"ticker": found_tickers[0], "tickers": found_tickers, "shock": shock, "date": date}


def infer_model_from_query(query: str) -> str:
    lowered = query.lower()
    if "linear threshold" in lowered or "linear_threshold" in lowered:
        return "linear_threshold"
    if "cascade removal" in lowered or "cascade_removal" in lowered:
        return "cascade_removal"
    return "debtrank"


def normalize_chat_query(query: str) -> str:
    """Normalize user query text for cleaner UI/logging."""
    normalized = " ".join(query.strip().split())
    if len(normalized) >= 2 and (
        (normalized[0] == normalized[-1] == '"')
        or (normalized[0] == normalized[-1] == "'")
    ):
        normalized = normalized[1:-1].strip()
    return normalized


def is_complex_query(query: str) -> bool:
    """Heuristic: detect prompts that need comparative/strategic GPT reasoning."""
    lowered = query.lower()
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
    return any(signal in lowered for signal in signals)


def is_compare_query(query: str, parsed: dict | None) -> bool:
    if not parsed or len(parsed.get("tickers", [])) < 2:
        return False
    lowered = query.lower()
    return any(signal in lowered for signal in ["compare", "rank", "difference", "versus", " vs ", "between"])


def is_query_in_scope(query: str, parsed: dict | None) -> tuple[bool, str]:
    """Guardrail: keep assistant focused on network/crisis/contagion scope."""
    if parsed:
        return True, "Parsed ticker/shock scenario."

    lowered = query.lower()
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
    if any(signal in lowered for signal in scope_signals):
        return True, "Detected network/crisis/contagion intent."
    return False, "Out of scope for RiskSentinel domain."


def tokenize_query(query: str, stopwords: set[str]) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{2,}", query.lower())
        if token not in stopwords
    }


def jaccard_similarity(a: set[str], b: set[str]) -> float:
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
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return None


def parse_structured_agent_output(text: str, schema_version: str) -> dict | None:
    payload = extract_json_payload(text)
    if not payload:
        return None

    required = ["situation", "quant_results", "risk_rating", "actions", "monitoring_triggers"]
    if not all(key in payload for key in required):
        return None

    def _as_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    parsed = {
        "schema_version": str(payload.get("schema_version", schema_version)),
        "situation": _as_list(payload.get("situation")),
        "quant_results": _as_list(payload.get("quant_results")),
        "risk_rating": str(payload.get("risk_rating", "UNKNOWN")).upper(),
        "actions": _as_list(payload.get("actions")),
        "monitoring_triggers": _as_list(payload.get("monitoring_triggers")),
        "evidence_used": _as_list(payload.get("evidence_used")),
        "notes": str(payload.get("notes", "")).strip(),
        "insufficient_data": bool(payload.get("insufficient_data", False)),
        "uncertainty_score": (
            float(payload.get("uncertainty_score", 0.5))
            if str(payload.get("uncertainty_score", "")).strip() != ""
            else 0.5
        ),
        "confidence_reason": str(payload.get("confidence_reason", "")).strip(),
        "validation": payload.get("validation", {})
        if isinstance(payload.get("validation"), dict)
        else {},
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
