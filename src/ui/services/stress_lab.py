"""Service helpers for the Stress Lab tab."""

from __future__ import annotations

import pandas as pd


def build_compare_rows_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [
        "rank",
        "ticker",
        "cascade_depth",
        "n_affected",
        "n_defaulted",
        "total_stress",
        "avg_stress_pct",
        "top_sectors",
    ]
    cols = [col for col in keep if col in df.columns]
    return df[cols].copy()
