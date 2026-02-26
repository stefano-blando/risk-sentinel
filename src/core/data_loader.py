"""
RiskSentinel — Data Loader
Loads pre-computed data from the PhD topological-stock-prediction project.
No recomputation needed: correlation matrices, centralities, and network
metrics are all pre-built (3081 daily snapshots, 210 S&P 500 stocks).
"""

import pickle
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS — PhD project data (final_version = clean thesis copy)
# ---------------------------------------------------------------------------
_LEGACY_PHD_BASE = Path.home() / "Scrivania/PHD/research/active/topological-stock-prediction/CESMA THESIS/network_stock_prediction"


def _candidate_processed_dirs() -> list[Path]:
    """Return candidate 'processed' dirs in priority order."""
    candidates: list[Path] = []
    env_root = os.getenv("RISKSENTINEL_DATA_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser()
        candidates.extend([
            root,
            root / "processed",
            root / "data" / "processed",
            root / "final_version" / "data" / "processed",
        ])

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root / "data" / "processed",
        _LEGACY_PHD_BASE / "final_version" / "data" / "processed",
        _LEGACY_PHD_BASE / "data" / "processed",
    ])

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _is_valid_processed_dir(path: Path) -> bool:
    return (path / "sector_mapping.parquet").is_file() and (path / "networks" / "node_centralities.pkl").is_file()


def _resolve_processed_dir() -> Path:
    candidates = _candidate_processed_dirs()
    first_existing: Path | None = None
    for path in candidates:
        if path.exists() and first_existing is None:
            first_existing = path
        if _is_valid_processed_dir(path):
            return path
    if first_existing is not None:
        return first_existing
    return candidates[0]


FINAL = _resolve_processed_dir()
NETWORKS = FINAL / "networks"
PHD_BASE = FINAL.parents[2] if len(FINAL.parents) >= 3 else _LEGACY_PHD_BASE


def get_data_root_info() -> dict[str, str]:
    """Expose resolved data paths for diagnostics/UI."""
    return {
        "final": str(FINAL),
        "networks": str(NETWORKS),
        "env_data_root": os.getenv("RISKSENTINEL_DATA_ROOT", ""),
    }


# ---------------------------------------------------------------------------
# SECTOR / TICKER METADATA
# ---------------------------------------------------------------------------
SECTOR_COLORS = {
    "Information Technology": "#1f77b4",
    "Health Care": "#ff7f0e",
    "Financials": "#2ca02c",
    "Consumer Discretionary": "#d62728",
    "Communication Services": "#9467bd",
    "Industrials": "#8c564b",
    "Consumer Staples": "#e377c2",
    "Energy": "#7f7f7f",
    "Utilities": "#bcbd22",
    "Real Estate": "#17becf",
    "Materials": "#aec7e8",
}

CRISIS_EVENTS = {
    "China Crisis 2015": ("2015-08-11", "2016-02-11"),
    "Brexit 2016": ("2016-06-23", "2016-07-08"),
    "Volmageddon 2018": ("2018-02-02", "2018-02-14"),
    "Q4 2018 Selloff": ("2018-10-01", "2018-12-31"),
    "COVID-19 Crash": ("2020-02-20", "2020-04-07"),
    "Russia-Ukraine 2022": ("2022-02-24", "2022-03-15"),
    "Rate Hike 2022": ("2022-09-01", "2022-10-31"),
    "SVB Crisis 2023": ("2023-03-08", "2023-03-24"),
    "Japan Carry Trade 2024": ("2024-08-01", "2024-08-15"),
}


# ---------------------------------------------------------------------------
# LIGHTWEIGHT LOADERS (parquet — fast, <100 MB total)
# ---------------------------------------------------------------------------
def load_sector_mapping() -> pd.DataFrame:
    """Ticker → Sector mapping (210 stocks, 11 GICS sectors)."""
    return pd.read_parquet(FINAL / "sector_mapping.parquet")


def get_sector_dict() -> dict[str, str]:
    """Returns {ticker: sector} dict."""
    sm = load_sector_mapping()
    return dict(zip(sm["Ticker"], sm["Sector"]))


def get_ticker_list() -> list[str]:
    """Returns sorted list of 210 tickers."""
    sm = load_sector_mapping()
    return sorted(sm["Ticker"].tolist())


def load_close_prices() -> pd.DataFrame:
    """Daily adjusted close prices (3140 days × 210 stocks)."""
    return pd.read_parquet(FINAL / "close_prices.parquet")


def load_returns() -> pd.DataFrame:
    """Daily simple returns (3140 × 210)."""
    return pd.read_parquet(FINAL / "returns_simple.parquet")


def load_market_data() -> pd.DataFrame:
    """VIX, SP500, SP500_Return (3140 rows)."""
    return pd.read_parquet(FINAL / "market_data.parquet")


def load_regime_data() -> pd.DataFrame:
    """VIX + regime labels (Calm/Normal/Elevated/High/Crisis) + CSAD."""
    return pd.read_parquet(FINAL / "regime_data.parquet")


def load_network_metrics() -> pd.DataFrame:
    """Global graph metrics time series (3081 × 13).
    Columns: nodes, edges, density, avg_degree, avg_clustering,
    n_components, largest_cc, largest_cc_pct, avg_weight, avg_abs_weight,
    method, vix, regime.
    """
    return pd.read_parquet(FINAL / "network_metrics.parquet")


def load_network_features() -> pd.DataFrame:
    """Full topological feature matrix (3081 × 182).
    Includes global metrics, centrality aggregates, delta features.
    """
    return pd.read_parquet(FINAL / "network_features.parquet")


def load_sector_centralities() -> pd.DataFrame:
    """Per-sector centrality averages (3081 × 55)."""
    return pd.read_parquet(FINAL / "sector_centrality_features.parquet")


# ---------------------------------------------------------------------------
# HEAVY LOADERS (pickle — load only when needed)
# ---------------------------------------------------------------------------
_corr_cache: Optional[dict] = None
_centrality_cache: Optional[dict] = None


def load_correlation_matrices() -> dict[pd.Timestamp, pd.DataFrame]:
    """Pre-computed rolling 60-day Pearson correlation matrices.
    Returns dict[Timestamp → DataFrame(210×210)].
    3081 daily snapshots, 2013-09-06 to 2025-12-04.
    WARNING: ~1 GB on disk, ~3-4 GB in RAM. Cached after first load.
    """
    global _corr_cache
    if _corr_cache is None:
        with open(NETWORKS / "correlation_matrices_pearson.pkl", "rb") as f:
            _corr_cache = pickle.load(f)
    return _corr_cache


def load_node_centralities() -> dict[pd.Timestamp, dict[str, dict[str, float]]]:
    """Pre-computed per-node centralities.
    Returns dict[Timestamp → {ticker: {degree, betweenness, closeness,
    eigenvector, pagerank}}].
    3081 dates × 210 tickers. 38 MB — fast to load. Cached.
    """
    global _centrality_cache
    if _centrality_cache is None:
        with open(NETWORKS / "node_centralities.pkl", "rb") as f:
            _centrality_cache = pickle.load(f)
    return _centrality_cache


# ---------------------------------------------------------------------------
# DATE UTILITIES
# ---------------------------------------------------------------------------
def get_available_dates() -> list[pd.Timestamp]:
    """Returns sorted list of dates with network data available."""
    centralities = load_node_centralities()
    return sorted(centralities.keys())


def find_nearest_date(target: str, dates: Optional[list] = None) -> pd.Timestamp:
    """Find the nearest available date to target (string or Timestamp)."""
    if dates is None:
        dates = get_available_dates()
    ts = pd.Timestamp(target)
    return min(dates, key=lambda d: abs(d - ts))


def get_correlation_matrix(date: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Get 210×210 correlation matrix for a specific date (nearest match).
    Returns (corr_matrix, actual_date).
    """
    corr_matrices = load_correlation_matrices()
    ts = pd.Timestamp(date)
    if ts in corr_matrices:
        return corr_matrices[ts], ts
    actual = find_nearest_date(date, sorted(corr_matrices.keys()))
    return corr_matrices[actual], actual


def get_node_centralities_for_date(date: str) -> tuple[dict[str, dict[str, float]], pd.Timestamp]:
    """Get per-node centralities for a specific date (nearest match).
    Returns ({ticker: {metric: value}}, actual_date).
    """
    centralities = load_node_centralities()
    ts = pd.Timestamp(date)
    if ts in centralities:
        return centralities[ts], ts
    actual = find_nearest_date(date, sorted(centralities.keys()))
    return centralities[actual], actual


def centralities_to_dataframe(node_centralities: dict) -> pd.DataFrame:
    """Convert single-date centrality dict to DataFrame.
    Input: {ticker: {degree, betweenness, closeness, eigenvector, pagerank}}
    Output: DataFrame with ticker as index, metrics as columns.
    """
    return pd.DataFrame.from_dict(node_centralities, orient="index")


# ---------------------------------------------------------------------------
# CONVENIENCE: LOAD MVP DATASET
# ---------------------------------------------------------------------------
def load_mvp_data() -> dict:
    """Load lightweight data for MVP (no 1 GB correlation pickle).
    Returns dict with: sector_mapping, market, regimes, network_metrics,
    network_features, node_centralities.
    Total: ~50 MB RAM.
    """
    return {
        "sector_mapping": load_sector_mapping(),
        "sector_dict": get_sector_dict(),
        "tickers": get_ticker_list(),
        "market": load_market_data(),
        "regimes": load_regime_data(),
        "network_metrics": load_network_metrics(),
        "network_features": load_network_features(),
        "node_centralities": load_node_centralities(),
    }
