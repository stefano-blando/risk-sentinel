"""Validation helpers for evidence references in structured payloads."""

from __future__ import annotations

import re
from typing import Any

REFERENCE_RE = re.compile(r"\b([ER]\d+)\b")
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def extract_references(text: str) -> set[str]:
    if not text:
        return set()
    return set(REFERENCE_RE.findall(text))


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def collect_payload_text(payload: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    for key in ("situation", "quant_results", "actions", "monitoring_triggers"):
        chunks.extend(_to_list(payload.get(key)))
    for key in ("notes", "confidence_reason", "risk_rating"):
        val = str(payload.get(key, "")).strip()
        if val:
            chunks.append(val)
    return chunks


def parse_evidence_used(entries: Any) -> tuple[set[str], list[str]]:
    refs: set[str] = set()
    invalid_entries: list[str] = []
    for raw in _to_list(entries):
        found = extract_references(raw)
        if not found:
            invalid_entries.append(raw)
            continue
        refs.update(found)
    return refs, invalid_entries


def validate_payload_evidence(
    payload: dict[str, Any],
    *,
    allowed_r_refs: set[str] | None = None,
    allowed_e_refs: set[str] | None = None,
    require_reference_for_numeric_claims: bool = False,
    facts_available: bool = True,
) -> dict[str, Any]:
    """Validate evidence references and basic claim-to-evidence constraints."""
    issues: list[str] = []
    required_fixes: list[str] = []

    evidence_refs, invalid_entries = parse_evidence_used(payload.get("evidence_used"))
    if invalid_entries:
        issues.append("evidence_used contains entries without E#/R# references.")
        required_fixes.append("Use only E# or R# tokens in evidence_used.")

    text_chunks = collect_payload_text(payload)
    cited_refs: set[str] = set()
    for text in text_chunks:
        cited_refs.update(extract_references(text))

    missing_refs = sorted(cited_refs - evidence_refs)
    if missing_refs:
        issues.append(f"Referenced citations not listed in evidence_used: {', '.join(missing_refs)}.")
        required_fixes.append("Include every cited E#/R# token in evidence_used.")

    if allowed_r_refs is not None:
        bad_r = sorted(ref for ref in evidence_refs if ref.startswith("R") and ref not in allowed_r_refs)
        if bad_r:
            issues.append(f"Unknown RAG references used: {', '.join(bad_r)}.")
            required_fixes.append("Use only retrieved R# references from current run.")

    if allowed_e_refs is not None:
        bad_e = sorted(ref for ref in evidence_refs if ref.startswith("E") and ref not in allowed_e_refs)
        if bad_e:
            issues.append(f"Unknown deterministic evidence references used: {', '.join(bad_e)}.")
            required_fixes.append("Use only E# references present in current evidence ledger.")

    has_numeric_claim = any(NUMERIC_RE.search(text) for text in text_chunks)
    insufficient = bool(payload.get("insufficient_data", False))
    if (
        require_reference_for_numeric_claims
        and has_numeric_claim
        and facts_available
        and not insufficient
        and not evidence_refs
    ):
        issues.append("Numeric claims are present without evidence references.")
        required_fixes.append("Add supporting E#/R# references in evidence_used for numeric claims.")

    return {
        "approved": len(issues) == 0,
        "issues": issues,
        "required_fixes": required_fixes,
        "evidence_refs": sorted(evidence_refs),
        "cited_refs": sorted(cited_refs),
        "has_numeric_claim": has_numeric_claim,
    }

