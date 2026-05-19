"""Evaluation helpers for control-plane architecture quality KPIs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSample:
    critic_approved: bool
    factual_consistent: bool
    latency_sec: float
    fallback_used: bool


def evaluate_samples(samples: list[EvalSample]) -> dict:
    n = len(samples)
    if n == 0:
        return {
            "n": 0,
            "critic_pass_rate_pct": 0.0,
            "factual_consistency_pct": 0.0,
            "latency_p95_sec": 0.0,
            "fallback_rate_pct": 0.0,
        }

    latencies = sorted(max(0.0, float(s.latency_sec)) for s in samples)
    idx = min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
    critic_pass = sum(1 for s in samples if s.critic_approved)
    factual_ok = sum(1 for s in samples if s.factual_consistent)
    fallback = sum(1 for s in samples if s.fallback_used)

    return {
        "n": n,
        "critic_pass_rate_pct": round(critic_pass / n * 100, 1),
        "factual_consistency_pct": round(factual_ok / n * 100, 1),
        "latency_p95_sec": round(latencies[idx], 3),
        "fallback_rate_pct": round(fallback / n * 100, 1),
    }
