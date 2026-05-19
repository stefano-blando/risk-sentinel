"""Service helpers for the Surveillance tab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_severity_df(result, sector_dict: dict[str, str] | None) -> pd.DataFrame:
    sector_lookup = sector_dict or {}
    sector_data: dict[str, dict[str, object]] = {}
    for node, stress in result.node_stress.items():
        sector = sector_lookup.get(node, "Unknown")
        if sector not in sector_data:
            sector_data[sector] = {"stresses": [], "defaulted": 0}
        sector_data[sector]["stresses"].append(stress)
        if stress >= 1.0:
            sector_data[sector]["defaulted"] += 1
    rows = []
    for sector, values in sector_data.items():
        stresses = values["stresses"]
        avg = float(np.mean(stresses))
        hit = sum(1 for stress in stresses if stress > 0.01)
        rows.append(
            {
                "Sector": sector,
                "Avg Stress %": round(avg * 100, 1),
                "Nodes Hit": hit,
                "Total": len(stresses),
                "Defaulted": int(values["defaulted"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Sector", "Avg Stress %", "Nodes Hit", "Total", "Defaulted"])
    return pd.DataFrame(rows).sort_values("Avg Stress %", ascending=False)


def compute_systemic_risk_index(result, total_nodes: int) -> tuple[float, str]:
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
