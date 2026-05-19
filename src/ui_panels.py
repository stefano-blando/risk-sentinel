"""UI panel helpers for explainability and KPI badges."""

from __future__ import annotations


def critic_verdict_badge(critic_approved: bool | None, critic_rounds: int | None) -> tuple[str, str]:
    if critic_approved is True:
        rounds = int(critic_rounds or 1)
        return (f"PASS ({rounds} round{'s' if rounds != 1 else ''})", "#16a34a")
    if critic_approved is False:
        return ("FAIL", "#dc2626")
    return ("N/A", "#64748b")


def critic_badge_html(critic_approved: bool | None, critic_rounds: int | None) -> str:
    label, color = critic_verdict_badge(critic_approved, critic_rounds)
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{color};color:white;font-size:12px;font-weight:600;'>Critic {label}</span>"
    )


def stage_badge_html(stage: str, status: str) -> str:
    status_u = str(status or "N/A").upper()
    color = {
        "PASS": "#16a34a",
        "FAIL": "#dc2626",
        "RUN": "#0ea5e9",
        "N/A": "#64748b",
    }.get(status_u, "#64748b")
    return (
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{color};color:white;font-size:12px;font-weight:600;'>"
        f"{stage} {status_u}</span>"
    )


def business_kpi_formula_markdown(kpi: dict | None) -> str:
    if not kpi:
        return ""
    return (
        "**Business KPI formula**\n"
        f"`{kpi.get('formula', 'avoided = expected_stress * coverage * efficiency')}`\n\n"
        f"Expected stress: **{kpi.get('expected_stress_pct', 0.0):.2f}%** | "
        f"Coverage: **{kpi.get('hedge_coverage_pct', 0.0):.2f}%** | "
        f"Efficiency: **{kpi.get('hedge_efficiency_pct', 0.0):.2f}%** | "
        f"Avoided: **{kpi.get('estimated_loss_avoided_pct', 0.0):.2f}%**"
    )
