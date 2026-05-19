"""Helpers to extend RiskSentinel processed datasets with newer market data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


REGIME_ORDER = ["Calm", "Normal", "Elevated", "High", "Crisis"]
CENTRALITY_COLUMNS = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
NETWORK_FEATURE_LAG_COLUMNS = [
    "density",
    "avg_degree",
    "avg_clustering",
    "largest_cc_pct",
    "avg_weight",
    "avg_abs_weight",
    "n_components",
    "n_communities",
    "modularity",
    "assortativity",
]


def classify_regime(vix: pd.Series) -> pd.Series:
    bins = [-np.inf, 16.0, 20.0, 25.0, 32.0, np.inf]
    cat = pd.cut(vix.astype(float), bins=bins, labels=REGIME_ORDER, right=False)
    return cat.astype(str)


def build_regime_frame(
    returns_simple: pd.DataFrame,
    sp500_close: pd.Series,
    vix_close: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build market_data + regime_data from aligned close/return series."""
    common_index = returns_simple.index
    sp500_close = sp500_close.reindex(common_index).ffill()
    vix_close = vix_close.reindex(common_index).ffill()

    sp500_return = sp500_close.pct_change().fillna(0.0)
    regime = classify_regime(vix_close)
    regime_numeric = regime.map({name: idx for idx, name in enumerate(REGIME_ORDER)}).astype(int)
    market_return = returns_simple.mean(axis=1).astype(float)
    csad = returns_simple.abs().mean(axis=1).astype(float)
    cs_std = returns_simple.std(axis=1).astype(float)

    market_data = pd.DataFrame(
        {
            "VIX": vix_close.astype(float),
            "SP500": sp500_close.astype(float),
            "SP500_Return": sp500_return.astype(float),
        },
        index=common_index,
    )
    market_data.index.name = "Date"

    regime_data = market_data.copy()
    regime_data["Regime"] = regime
    regime_data["Regime_Numeric"] = regime_numeric
    regime_data["HighVol"] = regime_numeric.ge(2).astype(int)
    regime_data["Crisis"] = regime.eq("Crisis").astype(int)
    regime_data["CSAD"] = csad
    regime_data["CS_Std"] = cs_std
    regime_data["Market_Return"] = market_return
    regime_data.index.name = "Date"
    return market_data, regime_data


def _safe_float(value: float | int | np.floating | np.integer | None) -> float:
    if value is None:
        return 0.0
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def _series_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return {
            f"avg_{prefix}": 0.0,
            f"std_{prefix}": 0.0,
            f"max_{prefix}": 0.0,
            f"min_{prefix}": 0.0,
            f"skew_{prefix}": 0.0,
        }
    return {
        f"avg_{prefix}": _safe_float(series.mean()),
        f"std_{prefix}": _safe_float(series.std(ddof=0)),
        f"max_{prefix}": _safe_float(series.max()),
        f"min_{prefix}": _safe_float(series.min()),
        f"skew_{prefix}": _safe_float(series.skew()),
    }


def _build_graph(corr: pd.DataFrame, threshold: float) -> nx.Graph:
    tickers = list(corr.columns)
    graph = nx.Graph()
    graph.add_nodes_from(tickers)
    values = corr.values
    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            weight = _safe_float(values[i, j])
            abs_weight = abs(weight)
            if abs_weight >= threshold:
                graph.add_edge(tickers[i], tickers[j], weight=weight, abs_weight=abs_weight)
    return graph


def _compute_snapshot_rows(
    date_ts: pd.Timestamp,
    corr: pd.DataFrame,
    graph: nx.Graph,
    vix: float,
    regime: str,
    sector_dict: dict[str, str],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float | str | pd.Timestamp],
    dict[str, float | pd.Timestamp],
    dict[str, float | str | pd.Timestamp],
]:
    degree = nx.degree_centrality(graph)
    n_nodes = graph.number_of_nodes()
    k_sample = min(n_nodes, max(1, min(20, max(1, n_nodes - 1))))
    betweenness = nx.betweenness_centrality(graph, k=k_sample, seed=42)
    closeness = nx.closeness_centrality(graph)
    pagerank = nx.pagerank(graph, weight="abs_weight") if graph.number_of_edges() else {n: 0.0 for n in graph.nodes()}
    try:
        eigenvector = (
            nx.eigenvector_centrality_numpy(graph, weight="abs_weight")
            if graph.number_of_edges()
            else {n: 0.0 for n in graph.nodes()}
        )
    except Exception:
        eigenvector = {n: 0.0 for n in graph.nodes()}

    node_centralities = {
        ticker: {
            "degree": _safe_float(degree.get(ticker, 0.0)),
            "betweenness": _safe_float(betweenness.get(ticker, 0.0)),
            "closeness": _safe_float(closeness.get(ticker, 0.0)),
            "eigenvector": _safe_float(eigenvector.get(ticker, 0.0)),
            "pagerank": _safe_float(pagerank.get(ticker, 0.0)),
        }
        for ticker in corr.columns
    }

    comps = list(nx.connected_components(graph))
    largest_cc = max((len(c) for c in comps), default=0)
    abs_weights = pd.Series([attrs.get("abs_weight", 0.0) for _, _, attrs in graph.edges(data=True)], dtype=float)
    signed_weights = pd.Series([attrs.get("weight", 0.0) for _, _, attrs in graph.edges(data=True)], dtype=float)
    degrees_raw = pd.Series([deg for _, deg in graph.degree()], dtype=float)

    if graph.number_of_edges():
        try:
            communities = list(nx.community.greedy_modularity_communities(graph, weight="abs_weight"))
            modularity = _safe_float(nx.community.modularity(graph, communities, weight="abs_weight"))
        except Exception:
            communities = [set(graph.nodes())]
            modularity = 0.0
        try:
            assortativity = _safe_float(nx.degree_pearson_correlation_coefficient(graph))
        except Exception:
            assortativity = 0.0
    else:
        communities = [{node} for node in graph.nodes()]
        modularity = 0.0
        assortativity = 0.0

    feature_row: dict[str, float | str | pd.Timestamp] = {
        "date": date_ts,
        "n_nodes": float(graph.number_of_nodes()),
        "n_edges": float(graph.number_of_edges()),
        "density": _safe_float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "avg_clustering": _safe_float(nx.average_clustering(graph, weight=None)) if graph.number_of_edges() else 0.0,
        "transitivity": _safe_float(nx.transitivity(graph)) if graph.number_of_edges() else 0.0,
        "assortativity": assortativity,
        "modularity": modularity,
        "n_communities": float(len(communities)),
        "n_components": float(len(comps)),
        "largest_cc_pct": _safe_float((largest_cc / graph.number_of_nodes()) * 100.0) if graph.number_of_nodes() else 0.0,
        "avg_weight": _safe_float(signed_weights.mean()),
        "avg_abs_weight": _safe_float(abs_weights.mean()),
        "weight_std": _safe_float(signed_weights.std(ddof=0)),
        "avg_degree": _safe_float(degrees_raw.mean()),
        "std_degree": _safe_float(degrees_raw.std(ddof=0)),
        "max_degree": _safe_float(degrees_raw.max()),
        "min_degree": _safe_float(degrees_raw.min()),
        "vix": _safe_float(vix),
    }
    feature_row.update(_series_stats(degrees_raw.tolist(), "degree"))
    for col in CENTRALITY_COLUMNS:
        feature_row.update(_series_stats([vals[col] for vals in node_centralities.values()], col))

    centrality_df = pd.DataFrame.from_dict(node_centralities, orient="index")
    centrality_df["Sector"] = centrality_df.index.map(sector_dict)
    grouped = centrality_df.groupby("Sector")[CENTRALITY_COLUMNS].mean(numeric_only=True)
    sector_row: dict[str, float | pd.Timestamp] = {"date": date_ts}
    for sector_name in sorted(set(sector_dict.values())):
        if sector_name not in grouped.index:
            continue
        sector_values = grouped.loc[sector_name]
        prefix = f"sect_{sector_name.lower().replace(' ', '_')}"
        for col in CENTRALITY_COLUMNS:
            sector_row[f"{prefix}_{col}"] = _safe_float(sector_values[col])

    metrics_row = {
        "date": date_ts,
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "density": feature_row["density"],
        "avg_degree": feature_row["avg_degree"],
        "avg_clustering": feature_row["avg_clustering"],
        "n_components": int(len(comps)),
        "largest_cc": int(largest_cc),
        "largest_cc_pct": feature_row["largest_cc_pct"],
        "avg_weight": feature_row["avg_weight"],
        "avg_abs_weight": feature_row["avg_abs_weight"],
        "method": "threshold",
        "vix": _safe_float(vix),
        "regime": str(regime),
    }

    return node_centralities, feature_row, sector_row, metrics_row


def compute_incremental_network_outputs(
    returns_simple: pd.DataFrame,
    market_data: pd.DataFrame,
    regime_data: pd.DataFrame,
    sector_dict: dict[str, str],
    existing_corr_matrices: dict[pd.Timestamp, pd.DataFrame],
    existing_node_centralities: dict[pd.Timestamp, dict[str, dict[str, float]]],
    threshold: float = 0.3,
    window: int = 60,
) -> tuple[
    dict[pd.Timestamp, pd.DataFrame],
    dict[pd.Timestamp, dict[str, dict[str, float]]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compute only the missing daily snapshots after the last processed date."""
    if not existing_corr_matrices:
        raise ValueError("existing_corr_matrices cannot be empty")

    last_snapshot = max(existing_corr_matrices)
    all_dates = list(returns_simple.index)

    new_corr = dict(existing_corr_matrices)
    new_centralities = dict(existing_node_centralities)
    feature_rows: list[dict[str, float | str | pd.Timestamp]] = []
    sector_rows: list[dict[str, float | pd.Timestamp]] = []
    metric_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for date_ts in all_dates:
        if date_ts <= last_snapshot:
            continue
        pos = int(returns_simple.index.get_loc(date_ts))
        if pos < window - 1:
            continue
        window_df = returns_simple.iloc[pos - window + 1 : pos + 1]
        corr = window_df.corr().fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        corr.index.name = "Ticker"
        corr.columns.name = "Ticker"

        graph = _build_graph(corr, threshold=threshold)
        node_row, feature_row, sector_row, metric_row = _compute_snapshot_rows(
            date_ts=date_ts,
            corr=corr,
            graph=graph,
            vix=_safe_float(market_data.loc[date_ts, "VIX"]),
            regime=str(regime_data.loc[date_ts, "Regime"]),
            sector_dict=sector_dict,
        )
        new_corr[date_ts] = corr
        new_centralities[date_ts] = node_row
        feature_rows.append(feature_row)
        sector_rows.append(sector_row)
        metric_rows.append(metric_row)

    feature_df = pd.DataFrame(feature_rows).set_index("date").sort_index() if feature_rows else pd.DataFrame()
    sector_df = pd.DataFrame(sector_rows).set_index("date").sort_index() if sector_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows).set_index("date").sort_index() if metric_rows else pd.DataFrame()
    return new_corr, new_centralities, feature_df, sector_df, metrics_df


def build_network_features(
    feature_core: pd.DataFrame,
    sector_centrality_features: pd.DataFrame,
) -> pd.DataFrame:
    if feature_core.empty:
        return feature_core.copy()

    features = feature_core.sort_index().copy()
    for base_col in NETWORK_FEATURE_LAG_COLUMNS:
        if base_col not in features.columns:
            continue
        for lag in (1, 5, 10, 20):
            prev = features[base_col].shift(lag)
            delta = features[base_col] - prev
            pct = (delta / prev.replace(0.0, np.nan)) * 100.0
            features[f"{base_col}_delta_{lag}d"] = delta.fillna(0.0).astype(float)
            features[f"{base_col}_pct_{lag}d"] = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    features["systemic_pressure"] = (features["density"] * features["vix"]).astype(float)
    if not sector_centrality_features.empty:
        overlap = [col for col in sector_centrality_features.columns if col in features.columns]
        if overlap:
            features = features.drop(columns=overlap)
        features = features.join(sector_centrality_features, how="left")

    features.index.name = "date"
    return features.sort_index()


def write_processed_dataset(
    output_dir: Path,
    sector_mapping: pd.DataFrame,
    close_prices: pd.DataFrame,
    returns_simple: pd.DataFrame,
    market_data: pd.DataFrame,
    regime_data: pd.DataFrame,
    network_metrics: pd.DataFrame,
    network_features: pd.DataFrame,
    sector_centrality_features: pd.DataFrame,
    correlation_matrices: dict[pd.Timestamp, pd.DataFrame],
    node_centralities: dict[pd.Timestamp, dict[str, dict[str, float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    networks_dir = output_dir / "networks"
    networks_dir.mkdir(parents=True, exist_ok=True)

    sector_mapping.to_parquet(output_dir / "sector_mapping.parquet")
    close_prices.to_parquet(output_dir / "close_prices.parquet")
    returns_simple.to_parquet(output_dir / "returns_simple.parquet")
    market_data.to_parquet(output_dir / "market_data.parquet")
    regime_data.to_parquet(output_dir / "regime_data.parquet")
    network_metrics.to_parquet(output_dir / "network_metrics.parquet")
    network_features.to_parquet(output_dir / "network_features.parquet")
    sector_centrality_features.to_parquet(output_dir / "sector_centrality_features.parquet")

    import pickle

    with open(networks_dir / "correlation_matrices_pearson.pkl", "wb") as fh:
        pickle.dump(correlation_matrices, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(networks_dir / "node_centralities.pkl", "wb") as fh:
        pickle.dump(node_centralities, fh, protocol=pickle.HIGHEST_PROTOCOL)
