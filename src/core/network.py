"""
RiskSentinel — Network Construction & Metrics
Builds NetworkX graphs from pre-computed correlation matrices and
provides metric computation functions for the agent tools.
"""

import networkx as nx
import numpy as np
import pandas as pd

from .data_loader import (
    SECTOR_COLORS,
    get_sector_dict,
    get_correlation_matrix,
    get_node_centralities_for_date,
    centralities_to_dataframe,
)


# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------
def build_network(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
    sector_dict: dict[str, str] | None = None,
) -> nx.Graph:
    """Build a NetworkX graph from a 210×210 correlation matrix.

    Edges: created where |correlation| > threshold.
    Edge attrs: weight (signed corr), abs_weight (|corr|).
    Node attrs: ticker, sector (if sector_dict provided).
    """
    if sector_dict is None:
        sector_dict = get_sector_dict()

    G = nx.Graph()

    # Add nodes with sector metadata
    for ticker in corr_matrix.columns:
        attrs = {"ticker": ticker}
        if ticker in sector_dict:
            attrs["sector"] = sector_dict[ticker]
            attrs["color"] = SECTOR_COLORS.get(sector_dict[ticker], "#cccccc")
        G.add_node(ticker, **attrs)

    # Add edges where |corr| > threshold (upper triangle only)
    tickers = corr_matrix.columns.tolist()
    values = corr_matrix.values
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr = values[i, j]
            if not np.isnan(corr) and abs(corr) > threshold:
                G.add_edge(
                    tickers[i], tickers[j],
                    weight=float(corr),
                    abs_weight=float(abs(corr)),
                )
    return G


def build_network_for_date(
    date: str,
    threshold: float = 0.3,
) -> tuple[nx.Graph, pd.Timestamp]:
    """Build a network for a specific date using pre-computed correlations.
    Returns (graph, actual_date_used).
    """
    corr_matrix, actual_date = get_correlation_matrix(date)
    G = build_network(corr_matrix, threshold=threshold)
    return G, actual_date


# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------
def compute_global_metrics(G: nx.Graph) -> dict:
    """Compute global network metrics for a graph."""
    if G.number_of_nodes() == 0:
        return {}

    metrics = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(d for _, d in G.degree()) / G.number_of_nodes(),
        "n_components": nx.number_connected_components(G),
    }

    # Metrics that need connected graph
    largest_cc = max(nx.connected_components(G), key=len)
    metrics["largest_cc_pct"] = len(largest_cc) / G.number_of_nodes()

    # Clustering (works on disconnected graphs)
    metrics["avg_clustering"] = nx.average_clustering(G)

    # Weighted metrics
    weights = [d["abs_weight"] for _, _, d in G.edges(data=True)]
    if weights:
        metrics["avg_weight"] = float(np.mean(weights))
        metrics["max_weight"] = float(np.max(weights))

    return metrics


def compute_node_centralities(G: nx.Graph) -> dict[str, dict[str, float]]:
    """Compute centrality metrics for all nodes.
    Returns {ticker: {degree, betweenness, closeness, eigenvector, pagerank}}.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, k=min(100, n))
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, weight="abs_weight")

    # Eigenvector can fail on disconnected graphs
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="abs_weight")
    except Exception:
        eigenvector = {node: 0.0 for node in G.nodes()}

    result = {}
    for node in G.nodes():
        result[node] = {
            "degree": degree[node],
            "betweenness": betweenness[node],
            "closeness": closeness[node],
            "eigenvector": eigenvector[node],
            "pagerank": pagerank[node],
        }
    return result


def get_top_nodes(
    centralities: dict[str, dict[str, float]],
    metric: str = "pagerank",
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Get top-N nodes by a centrality metric.
    Returns [(ticker, value), ...] sorted descending.
    """
    ranked = sorted(
        centralities.items(),
        key=lambda x: x[1].get(metric, 0),
        reverse=True,
    )
    return [(ticker, metrics[metric]) for ticker, metrics in ranked[:top_n]]


def get_node_neighbors(G: nx.Graph, ticker: str) -> list[tuple[str, float]]:
    """Get all neighbors of a node with edge weights.
    Returns [(neighbor_ticker, correlation), ...] sorted by |corr| descending.
    """
    if ticker not in G:
        return []
    neighbors = []
    for neighbor in G.neighbors(ticker):
        weight = G[ticker][neighbor].get("weight", 0)
        neighbors.append((neighbor, weight))
    return sorted(neighbors, key=lambda x: abs(x[1]), reverse=True)


def get_sector_subgraph(G: nx.Graph, sector: str) -> nx.Graph:
    """Extract subgraph for a specific GICS sector."""
    nodes = [n for n, d in G.nodes(data=True) if d.get("sector") == sector]
    return G.subgraph(nodes).copy()


# ---------------------------------------------------------------------------
# NETWORK COMPARISON (for what-if scenarios)
# ---------------------------------------------------------------------------
def compare_networks(G_before: nx.Graph, G_after: nx.Graph) -> dict:
    """Compare two network states (pre/post shock).
    Returns dict with delta metrics.
    """
    m_before = compute_global_metrics(G_before)
    m_after = compute_global_metrics(G_after)

    comparison = {}
    for key in m_before:
        if isinstance(m_before[key], (int, float)):
            comparison[f"{key}_before"] = m_before[key]
            comparison[f"{key}_after"] = m_after[key]
            comparison[f"{key}_delta"] = m_after[key] - m_before[key]

    return comparison
