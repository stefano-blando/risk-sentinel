"""
RiskSentinel — Data Loader
Loads pre-computed data from the PhD topological-stock-prediction project.
If those files are unavailable (for example on Streamlit Cloud), it can fall
back to a deterministic synthetic demo dataset so the app remains runnable.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATHS — PhD project data (final_version = clean thesis copy)
# ---------------------------------------------------------------------------
_LEGACY_PHD_BASE = Path.home() / "Scrivania/PHD/research/active/topological-stock-prediction/CESMA THESIS/network_stock_prediction"


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

REGIME_ORDER = ["Calm", "Normal", "Elevated", "High", "Crisis"]
REGIME_TO_NUMERIC = {name: idx for idx, name in enumerate(REGIME_ORDER)}
NUMERIC_TO_REGIME = {idx: name for name, idx in REGIME_TO_NUMERIC.items()}


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


def _allow_synthetic_data() -> bool:
    raw = os.getenv("RISKSENTINEL_ALLOW_SYNTHETIC_DATA", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def is_synthetic_mode() -> bool:
    """True when data loaders are using generated deterministic demo data."""
    return _allow_synthetic_data() and (not _is_valid_processed_dir(FINAL))


def get_data_root_info() -> dict[str, str]:
    """Expose resolved data paths and mode for diagnostics/UI."""
    return {
        "final": str(FINAL),
        "networks": str(NETWORKS),
        "env_data_root": os.getenv("RISKSENTINEL_DATA_ROOT", ""),
        "data_mode": "synthetic_demo" if is_synthetic_mode() else "processed_files",
        "synthetic_allowed": "true" if _allow_synthetic_data() else "false",
    }


def _demo_sector_universe() -> dict[str, list[str]]:
    return {
        "Information Technology": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM"],
        "Health Care": ["UNH", "JNJ", "PFE", "MRK", "ABBV"],
        "Financials": ["JPM", "GS", "BAC", "MS", "C", "WFC", "SCHW", "BLK"],
        "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T"],
        "Industrials": ["BA", "CAT", "GE", "HON", "UPS"],
        "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Utilities": ["NEE", "SO", "DUK", "EXC", "AEP"],
        "Real Estate": ["AMT", "PLD", "O", "EQIX", "SPG"],
        "Materials": ["LIN", "APD", "FCX", "NEM", "SHW"],
    }


def _classify_regime(vix: pd.Series) -> pd.Series:
    bins = [-np.inf, 16.0, 20.0, 25.0, 32.0, np.inf]
    labels = REGIME_ORDER
    cat = pd.cut(vix, bins=bins, labels=labels, right=False)
    return cat.astype(str)


def _normalize_regime_data(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()

    if "Regime" not in normalized.columns and "Regime_Numeric" in normalized.columns:
        numeric = pd.to_numeric(normalized["Regime_Numeric"], errors="coerce")
        normalized["Regime"] = numeric.round().map(NUMERIC_TO_REGIME)

    if "Regime_Numeric" not in normalized.columns:
        if "Regime" in normalized.columns:
            regime_labels = normalized["Regime"].astype(str).str.strip()
            numeric = regime_labels.map(REGIME_TO_NUMERIC)
            if numeric.isna().any() and "VIX" in normalized.columns:
                fallback_numeric = _classify_regime(pd.to_numeric(normalized["VIX"], errors="coerce")).map(REGIME_TO_NUMERIC)
                numeric = numeric.fillna(fallback_numeric)
            normalized["Regime_Numeric"] = numeric
        elif "VIX" in normalized.columns:
            inferred_regime = _classify_regime(pd.to_numeric(normalized["VIX"], errors="coerce"))
            normalized["Regime"] = inferred_regime
            normalized["Regime_Numeric"] = inferred_regime.map(REGIME_TO_NUMERIC)

    if "Regime" in normalized.columns:
        normalized["Regime"] = normalized["Regime"].astype(str).str.strip()
    if "Regime_Numeric" in normalized.columns:
        normalized["Regime_Numeric"] = pd.to_numeric(normalized["Regime_Numeric"], errors="coerce").round()
        normalized["Regime_Numeric"] = normalized["Regime_Numeric"].fillna(0).astype(int)

    if "Regime" not in normalized.columns and "Regime_Numeric" in normalized.columns:
        normalized["Regime"] = normalized["Regime_Numeric"].map(NUMERIC_TO_REGIME).fillna("Calm")
    elif "Regime" in normalized.columns and "Regime_Numeric" in normalized.columns:
        normalized["Regime"] = normalized["Regime"].replace({"nan": None}).fillna(
            normalized["Regime_Numeric"].map(NUMERIC_TO_REGIME)
        )

    if "HighVol" not in normalized.columns and "Regime_Numeric" in normalized.columns:
        normalized["HighVol"] = normalized["Regime_Numeric"].ge(2).astype(int)
    if "Crisis" not in normalized.columns and "Regime" in normalized.columns:
        normalized["Crisis"] = normalized["Regime"].eq("Crisis").astype(int)

    return normalized


def _normalize_network_features(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy().sort_index()
    if "density" not in normalized.columns:
        normalized["density"] = 0.0
    if "avg_clustering" not in normalized.columns:
        normalized["avg_clustering"] = 0.0
    if "avg_abs_weight" not in normalized.columns:
        if "abs_weight" in normalized.columns:
            normalized["avg_abs_weight"] = pd.to_numeric(normalized["abs_weight"], errors="coerce").abs()
        elif "avg_weight" in normalized.columns:
            normalized["avg_abs_weight"] = pd.to_numeric(normalized["avg_weight"], errors="coerce").abs()
        else:
            try:
                metrics = load_network_metrics()
            except Exception:
                metrics = pd.DataFrame(index=normalized.index)
            if "avg_abs_weight" in metrics.columns:
                aligned = metrics["avg_abs_weight"].reindex(normalized.index)
                normalized["avg_abs_weight"] = pd.to_numeric(aligned, errors="coerce")
            elif "avg_weight" in metrics.columns:
                aligned = metrics["avg_weight"].reindex(normalized.index)
                normalized["avg_abs_weight"] = pd.to_numeric(aligned, errors="coerce").abs()
            else:
                normalized["avg_abs_weight"] = 0.0

    if "vix" not in normalized.columns:
        try:
            regime = load_regime_data()
            if "VIX" in regime.columns:
                normalized["vix"] = pd.to_numeric(regime["VIX"].reindex(normalized.index), errors="coerce")
        except Exception:
            pass
    if "vix" not in normalized.columns:
        normalized["vix"] = 0.0

    normalized["density"] = pd.to_numeric(normalized["density"], errors="coerce")
    normalized["avg_clustering"] = pd.to_numeric(normalized["avg_clustering"], errors="coerce")
    normalized["avg_abs_weight"] = pd.to_numeric(normalized["avg_abs_weight"], errors="coerce")
    normalized["vix"] = pd.to_numeric(normalized["vix"], errors="coerce")
    normalized["density"] = normalized["density"].ffill().bfill().fillna(0.0)
    normalized["avg_clustering"] = normalized["avg_clustering"].ffill().bfill().fillna(0.0)
    normalized["avg_abs_weight"] = normalized["avg_abs_weight"].ffill().bfill().fillna(0.0)
    normalized["vix"] = normalized["vix"].ffill().bfill().fillna(0.0)
    return normalized


def _build_synthetic_dataset() -> dict:
    rng = np.random.default_rng(20260307)

    sector_universe = _demo_sector_universe()
    sectors = list(sector_universe.keys())
    tickers = [t for sec in sectors for t in sector_universe[sec]]
    n_tickers = len(tickers)

    sector_lookup = {ticker: sector for sector, names in sector_universe.items() for ticker in names}
    sector_idx = np.array([sectors.index(sector_lookup[t]) for t in tickers], dtype=int)

    all_dates = pd.bdate_range("2018-01-02", "2025-12-31")
    n_days = len(all_dates)

    # Factor model: market + sector + idiosyncratic components.
    cyc = 1.0 + 0.65 * np.maximum(0.0, np.sin(np.linspace(0, 18 * np.pi, n_days)))
    market_factor = rng.normal(0.0, 0.006, size=n_days) * cyc
    sector_factors = rng.normal(0.0, 0.0045, size=(n_days, len(sectors))) * cyc[:, None]
    idio = rng.normal(0.0, 0.0105, size=(n_days, n_tickers)) * (0.75 + 0.35 * cyc[:, None])

    beta_m = rng.uniform(0.70, 1.25, size=n_tickers)
    beta_s = rng.uniform(0.45, 1.05, size=n_tickers)

    returns_np = market_factor[:, None] * beta_m
    returns_np += sector_factors[:, sector_idx] * beta_s
    returns_np += idio
    returns_np = np.clip(returns_np, -0.22, 0.22)

    returns = pd.DataFrame(returns_np, index=all_dates, columns=tickers)

    base_prices = rng.uniform(40.0, 240.0, size=n_tickers)
    close_prices = (1.0 + returns).cumprod()
    close_prices = close_prices.mul(base_prices, axis=1)

    sp500_ret = (returns.mean(axis=1) + rng.normal(0.0, 0.0015, size=n_days)).clip(-0.12, 0.12)
    sp500 = 4200.0 * (1.0 + sp500_ret).cumprod()

    rolling_vol = sp500_ret.rolling(20, min_periods=5).std().bfill().fillna(0.01)
    vix = (11.0 + 950.0 * rolling_vol + 3.0 * np.maximum(0.0, np.sin(np.linspace(0, 12 * np.pi, n_days)))).clip(10.0, 55.0)
    regime = _classify_regime(vix)
    csad = returns.abs().mean(axis=1)

    market_data = pd.DataFrame(
        {
            "VIX": vix.astype(float),
            "SP500": sp500.astype(float),
            "SP500_Return": sp500_ret.astype(float),
        },
        index=all_dates,
    )

    regime_data = market_data.copy()
    regime_data["Regime"] = regime
    regime_data["CSAD"] = csad.astype(float)

    # Build monthly snapshots with 60-day rolling correlations.
    candidate_snapshots = pd.bdate_range(all_dates[80], all_dates[-1], freq="BMS")
    snapshot_dates: list[pd.Timestamp] = []
    corr_matrices: dict[pd.Timestamp, pd.DataFrame] = {}
    node_centralities: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    metrics_rows: list[dict] = []

    for d in candidate_snapshots:
        pos = int(all_dates.get_indexer([d], method="nearest")[0])
        if pos < 59:
            continue

        date_ts = pd.Timestamp(all_dates[pos])
        window = returns.iloc[pos - 59: pos + 1]
        corr = window.corr().fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)

        snapshot_dates.append(date_ts)
        corr_matrices[date_ts] = corr

        G = nx.Graph()
        for ticker in tickers:
            G.add_node(ticker)
        values = corr.values
        for i in range(n_tickers):
            for j in range(i + 1, n_tickers):
                w = float(values[i, j])
                abs_w = abs(w)
                if abs_w >= 0.35:
                    G.add_edge(tickers[i], tickers[j], weight=w, abs_weight=abs_w)

        degree = nx.degree_centrality(G)
        k_sample = max(5, min(20, max(1, G.number_of_nodes() - 1)))
        betweenness = nx.betweenness_centrality(G, k=k_sample, seed=42)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G, weight="abs_weight") if G.number_of_edges() else {n: 0.0 for n in G.nodes()}
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="abs_weight") if G.number_of_edges() else {n: 0.0 for n in G.nodes()}
        except Exception:
            eigenvector = {n: 0.0 for n in G.nodes()}

        node_centralities[date_ts] = {
            ticker: {
                "degree": float(degree.get(ticker, 0.0)),
                "betweenness": float(betweenness.get(ticker, 0.0)),
                "closeness": float(closeness.get(ticker, 0.0)),
                "eigenvector": float(eigenvector.get(ticker, 0.0)),
                "pagerank": float(pagerank.get(ticker, 0.0)),
            }
            for ticker in tickers
        }

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        abs_weights = [float(attrs.get("abs_weight", 0.0)) for _, _, attrs in G.edges(data=True)]
        signed_weights = [float(attrs.get("weight", 0.0)) for _, _, attrs in G.edges(data=True)]

        if n_nodes > 0:
            comps = list(nx.connected_components(G))
            largest_cc = max((len(c) for c in comps), default=0)
            n_components = len(comps)
            avg_degree = float(np.mean([deg for _, deg in G.degree()]))
            avg_clustering = float(nx.average_clustering(G)) if n_edges else 0.0
        else:
            largest_cc = 0
            n_components = 0
            avg_degree = 0.0
            avg_clustering = 0.0

        metrics_rows.append(
            {
                "date": date_ts,
                "nodes": int(n_nodes),
                "edges": int(n_edges),
                "density": float(nx.density(G)) if n_nodes > 1 else 0.0,
                "avg_degree": avg_degree,
                "avg_clustering": avg_clustering,
                "n_components": int(n_components),
                "largest_cc": int(largest_cc),
                "largest_cc_pct": float(largest_cc / n_nodes) if n_nodes else 0.0,
                "avg_weight": float(np.mean(signed_weights)) if signed_weights else 0.0,
                "avg_abs_weight": float(np.mean(abs_weights)) if abs_weights else 0.0,
                "method": "synthetic_demo",
                "vix": float(market_data.loc[date_ts, "VIX"]),
                "regime": str(regime_data.loc[date_ts, "Regime"]),
            }
        )

    network_metrics = pd.DataFrame(metrics_rows).set_index("date").sort_index()

    network_features = pd.DataFrame(index=network_metrics.index)
    network_features["density"] = network_metrics["density"]
    network_features["avg_abs_weight"] = network_metrics["avg_abs_weight"]
    network_features["avg_degree"] = network_metrics["avg_degree"]
    network_features["avg_clustering"] = network_metrics["avg_clustering"]
    network_features["largest_cc_pct"] = network_metrics["largest_cc_pct"]
    network_features["vix"] = network_metrics["vix"]
    network_features["density_delta_5"] = network_features["density"].diff(5).fillna(0.0)
    network_features["avg_degree_delta_5"] = network_features["avg_degree"].diff(5).fillna(0.0)
    network_features["systemic_pressure"] = (network_features["density"] * network_features["vix"]).astype(float)

    sector_mapping = pd.DataFrame(
        {"Ticker": tickers, "Sector": [sector_lookup[t] for t in tickers]}
    )

    sector_rows: list[dict] = []
    for date_ts in snapshot_dates:
        cdf = pd.DataFrame.from_dict(node_centralities[date_ts], orient="index")
        cdf["Sector"] = cdf.index.map(sector_lookup)
        grouped = cdf.groupby("Sector")[["degree", "betweenness", "closeness", "eigenvector", "pagerank"]].mean()
        row: dict[str, float | str | pd.Timestamp] = {"date": date_ts}
        for sector_name in sectors:
            if sector_name not in grouped.index:
                continue
            vals = grouped.loc[sector_name]
            prefix = sector_name.lower().replace(" ", "_")
            row[f"{prefix}_degree"] = float(vals["degree"])
            row[f"{prefix}_betweenness"] = float(vals["betweenness"])
            row[f"{prefix}_pagerank"] = float(vals["pagerank"])
        sector_rows.append(row)

    sector_centrality_features = pd.DataFrame(sector_rows).set_index("date").sort_index()

    return {
        "sector_mapping": sector_mapping,
        "close_prices": close_prices,
        "returns_simple": returns,
        "market_data": market_data,
        "regime_data": regime_data,
        "network_metrics": network_metrics,
        "network_features": network_features,
        "sector_centrality_features": sector_centrality_features,
        "correlation_matrices": corr_matrices,
        "node_centralities": node_centralities,
    }


_SYNTHETIC_CACHE: Optional[dict] = None


def _synthetic_data() -> dict:
    global _SYNTHETIC_CACHE
    if _SYNTHETIC_CACHE is None:
        _SYNTHETIC_CACHE = _build_synthetic_dataset()
    return _SYNTHETIC_CACHE


def _load_parquet_or_synthetic(filename: str, synthetic_key: str) -> pd.DataFrame:
    path = FINAL / filename
    if path.is_file():
        return pd.read_parquet(path)
    if is_synthetic_mode():
        return _synthetic_data()[synthetic_key].copy()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# LIGHTWEIGHT LOADERS (parquet — fast, <100 MB total)
# ---------------------------------------------------------------------------
def load_sector_mapping() -> pd.DataFrame:
    """Ticker → Sector mapping."""
    return _load_parquet_or_synthetic("sector_mapping.parquet", "sector_mapping")


def get_sector_dict() -> dict[str, str]:
    """Returns {ticker: sector} dict."""
    sm = load_sector_mapping()
    return dict(zip(sm["Ticker"], sm["Sector"]))


def get_ticker_list() -> list[str]:
    """Returns sorted ticker list."""
    sm = load_sector_mapping()
    return sorted(sm["Ticker"].tolist())


def load_close_prices() -> pd.DataFrame:
    """Daily adjusted close prices."""
    return _load_parquet_or_synthetic("close_prices.parquet", "close_prices")


def load_returns() -> pd.DataFrame:
    """Daily simple returns."""
    return _load_parquet_or_synthetic("returns_simple.parquet", "returns_simple")


def load_market_data() -> pd.DataFrame:
    """VIX, SP500, SP500_Return."""
    return _load_parquet_or_synthetic("market_data.parquet", "market_data")


def load_regime_data() -> pd.DataFrame:
    """VIX + regime labels + CSAD."""
    frame = _load_parquet_or_synthetic("regime_data.parquet", "regime_data")
    return _normalize_regime_data(frame)


def load_network_metrics() -> pd.DataFrame:
    """Global graph metrics time series."""
    return _load_parquet_or_synthetic("network_metrics.parquet", "network_metrics")


def load_network_features() -> pd.DataFrame:
    """Topological feature matrix."""
    frame = _load_parquet_or_synthetic("network_features.parquet", "network_features")
    return _normalize_network_features(frame)


def load_sector_centralities() -> pd.DataFrame:
    """Per-sector centrality averages."""
    return _load_parquet_or_synthetic("sector_centrality_features.parquet", "sector_centrality_features")


# ---------------------------------------------------------------------------
# HEAVY LOADERS (pickle — load only when needed)
# ---------------------------------------------------------------------------
_corr_cache: Optional[dict] = None
_centrality_cache: Optional[dict] = None


def load_correlation_matrices() -> dict[pd.Timestamp, pd.DataFrame]:
    """Pre-computed rolling correlation matrices."""
    global _corr_cache
    if _corr_cache is not None:
        return _corr_cache

    pkl_path = NETWORKS / "correlation_matrices_pearson.pkl"
    if pkl_path.is_file():
        with open(pkl_path, "rb") as f:
            _corr_cache = pickle.load(f)
        return _corr_cache

    if is_synthetic_mode():
        _corr_cache = _synthetic_data()["correlation_matrices"]
        return _corr_cache

    with open(pkl_path, "rb") as f:
        _corr_cache = pickle.load(f)
    return _corr_cache


def load_node_centralities() -> dict[pd.Timestamp, dict[str, dict[str, float]]]:
    """Pre-computed per-node centralities."""
    global _centrality_cache
    if _centrality_cache is not None:
        return _centrality_cache

    pkl_path = NETWORKS / "node_centralities.pkl"
    if pkl_path.is_file():
        with open(pkl_path, "rb") as f:
            _centrality_cache = pickle.load(f)
        return _centrality_cache

    if is_synthetic_mode():
        _centrality_cache = _synthetic_data()["node_centralities"]
        return _centrality_cache

    with open(pkl_path, "rb") as f:
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
    """Get correlation matrix for a specific date (nearest match)."""
    corr_matrices = load_correlation_matrices()
    ts = pd.Timestamp(date)
    if ts in corr_matrices:
        return corr_matrices[ts], ts
    actual = find_nearest_date(date, sorted(corr_matrices.keys()))
    return corr_matrices[actual], actual


def get_node_centralities_for_date(date: str) -> tuple[dict[str, dict[str, float]], pd.Timestamp]:
    """Get per-node centralities for a specific date (nearest match)."""
    centralities = load_node_centralities()
    ts = pd.Timestamp(date)
    if ts in centralities:
        return centralities[ts], ts
    actual = find_nearest_date(date, sorted(centralities.keys()))
    return centralities[actual], actual


def centralities_to_dataframe(node_centralities: dict) -> pd.DataFrame:
    """Convert single-date centrality dict to DataFrame."""
    return pd.DataFrame.from_dict(node_centralities, orient="index")


# ---------------------------------------------------------------------------
# CONVENIENCE: LOAD MVP DATASET
# ---------------------------------------------------------------------------
def load_mvp_data() -> dict:
    """Load lightweight data for MVP/no-recompute flows."""
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
