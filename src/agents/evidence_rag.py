"""Evidence-RAG helpers for RiskSentinel.

Retrieves compact, cited context blocks from:
- static crisis knowledge
- recent execution history
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "if", "then",
    "into", "using", "risk", "systemic", "compare", "between", "query", "plan",
}

CRISIS_BRIEFS = {
    "COVID-19": "Correlations surged market-wide; VIX reached extreme levels and cross-sector contagion accelerated.",
    "SVB": "Stress was concentrated in Financials with selective spillovers into rate-sensitive sectors.",
    "Japan Carry Trade": "Fast cross-asset repricing produced short-lived but broad correlation spikes.",
    "Volmageddon": "Volatility shock propagated rapidly through equity and derivatives-linked exposures.",
    "Russia-Ukraine": "Energy and macro uncertainty created abrupt sector dispersion and contagion channels.",
}


@dataclass(frozen=True)
class EvidenceDoc:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedEvidence:
    reference_id: str
    doc_id: str
    source: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any]


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z]{3,}", text.lower())
        if tok not in STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def build_crisis_evidence_docs(crisis_events: dict[str, tuple[str, str]]) -> list[EvidenceDoc]:
    docs: list[EvidenceDoc] = []
    items = sorted(crisis_events.items(), key=lambda kv: kv[1][0])
    for idx, (name, (start, end)) in enumerate(items, start=1):
        brief = ""
        for key, val in CRISIS_BRIEFS.items():
            if key.lower() in name.lower():
                brief = val
                break
        if not brief:
            brief = "Historical regime shift with elevated contagion risk and correlation re-pricing."
        docs.append(
            EvidenceDoc(
                doc_id=f"C{idx}",
                source="crisis_catalog",
                title=name,
                text=f"{name} ({start} to {end}). {brief}",
                metadata={"start": start, "end": end},
            )
        )
    return docs


def build_history_evidence_docs(history: list[dict], max_items: int = 12) -> list[EvidenceDoc]:
    docs: list[EvidenceDoc] = []
    rows = list(history)[-max_items:]
    for i, row in enumerate(reversed(rows), start=1):
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        result = row.get("result", {}) if isinstance(row.get("result", {}), dict) else {}
        timings = row.get("timings", {}) if isinstance(row.get("timings", {}), dict) else {}
        quality = row.get("quality", {}) if isinstance(row.get("quality", {}), dict) else {}
        state = str(result.get("state", "n/a"))
        summary = (
            f"Prior run: state={state}, latency={timings.get('total_sec', 'n/a')}, "
            f"factual={quality.get('factual_consistency', 'n/a')}, "
            f"uncertainty={quality.get('uncertainty_score', 'n/a')}"
        )
        docs.append(
            EvidenceDoc(
                doc_id=f"H{i}",
                source="run_history",
                title=query[:90],
                text=summary,
                metadata={"query": query, "state": state},
            )
        )
    return docs


def retrieve_evidence(
    query: str,
    docs: list[EvidenceDoc],
    top_k: int = 4,
    min_score: float = 0.03,
) -> list[RetrievedEvidence]:
    q_tokens = _tokenize(query)
    q_tickers = set(re.findall(r"\b[A-Z]{1,5}\b", query.upper()))
    q_dates = set(re.findall(r"\b20\d{2}\b", query))

    scored: list[tuple[float, EvidenceDoc]] = []
    for doc in docs:
        text = f"{doc.title} {doc.text}"
        d_tokens = _tokenize(text)
        score = _jaccard(q_tokens, d_tokens)

        upper_text = text.upper()
        if q_tickers and any(tk in upper_text for tk in q_tickers):
            score += 0.18
        if q_dates and any(dt in text for dt in q_dates):
            score += 0.10
        if doc.source == "crisis_catalog" and any(k in query.lower() for k in ["crisis", "regime", "historical"]):
            score += 0.08

        if score >= min_score:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[RetrievedEvidence] = []
    for rank, (score, doc) in enumerate(scored[: max(1, top_k)], start=1):
        out.append(
            RetrievedEvidence(
                reference_id=f"R{rank}",
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=doc.text,
                score=round(float(score), 4),
                metadata=doc.metadata,
            )
        )
    return out


def format_evidence_block(retrieved: list[RetrievedEvidence], max_chars: int = 1800) -> str:
    if not retrieved:
        return ""

    lines = ["Retrieved evidence (cite R# in evidence_used/notes when relevant):"]
    used = len(lines[0])
    for item in retrieved:
        line = (
            f"[{item.reference_id}] {item.title} ({item.source}) | "
            f"score={item.score:.2f} | {item.text}"
        )
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def serialize_retrieved(retrieved: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "reference_id": r.reference_id,
            "doc_id": r.doc_id,
            "source": r.source,
            "title": r.title,
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata,
        }
        for r in retrieved
    ]
