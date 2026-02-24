import sys
sys.path.insert(0, '.')

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.core import data_loader as dl
from src.core import network as nw


CENTRALITY_KEYS = {"degree", "betweenness", "closeness", "eigenvector", "pagerank"}
GLOBAL_METRIC_KEYS = {
    "n_nodes",
    "n_edges",
    "density",
    "avg_degree",
    "n_components",
    "largest_cc_pct",
    "avg_clustering",
    "avg_weight",
    "max_weight",
}


def _require_file(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"Missing required data file: {path}")


@pytest.fixture(scope="module")
def sample_corr_df() -> pd.DataFrame:
    tickers = ["A", "B", "C", "D", "E"]
    data = [
        [1.00, 0.60, 0.20, -0.40, 0.10],
        [0.60, 1.00, 0.31, 0.00, -0.29],
        [0.20, 0.31, 1.00, 0.45, 0.05],
        [-0.40, 0.00, 0.45, 1.00, -0.80],
        [0.10, -0.29, 0.05, -0.80, 1.00],
    ]
    return pd.DataFrame(data, index=tickers, columns=tickers)


@pytest.fixture(scope="module")
def sample_sector_dict() -> dict[str, str]:
    return {
        "A": "Tech",
        "B": "Financials",
        "C": "Financials",
        "D": "Industrials",
        "E": "Financials",
    }


@pytest.fixture(scope="module")
def real_graph_for_date() -> tuple[nx.Graph, pd.Timestamp]:
    _require_file(dl.NETWORKS / "correlation_matrices_pearson.pkl")
    _require_file(dl.FINAL / "sector_mapping.parquet")
    return nw.build_network_for_date("2025-12-01")


def test_build_network(sample_corr_df: pd.DataFrame, sample_sector_dict: dict[str, str]) -> None:
    graph = nw.build_network(sample_corr_df, threshold=0.3, sector_dict=sample_sector_dict)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 5
    # Expected edges where |corr| > 0.3: AB, AD, BC, CD, DE => 5 edges.
    assert graph.number_of_edges() == 5

    assert graph.has_edge("A", "B")
    assert graph.has_edge("D", "E")
    assert graph["A"]["B"]["weight"] == pytest.approx(0.60)
    assert graph["A"]["B"]["abs_weight"] == pytest.approx(0.60)
    assert graph["D"]["E"]["weight"] == pytest.approx(-0.80)
    assert graph["D"]["E"]["abs_weight"] == pytest.approx(0.80)


@pytest.mark.slow
def test_build_network_for_date(real_graph_for_date: tuple[nx.Graph, pd.Timestamp]) -> None:
    graph, actual_date = real_graph_for_date

    assert isinstance(graph, nx.Graph)
    assert isinstance(actual_date, pd.Timestamp)
    assert graph.number_of_nodes() == 210


def test_compute_global_metrics(sample_corr_df: pd.DataFrame, sample_sector_dict: dict[str, str]) -> None:
    graph = nw.build_network(sample_corr_df, threshold=0.3, sector_dict=sample_sector_dict)
    metrics = nw.compute_global_metrics(graph)

    assert isinstance(metrics, dict)
    assert GLOBAL_METRIC_KEYS.issubset(metrics.keys())


def test_compute_node_centralities(sample_corr_df: pd.DataFrame, sample_sector_dict: dict[str, str]) -> None:
    graph = nw.build_network(sample_corr_df, threshold=0.3, sector_dict=sample_sector_dict)
    centralities = nw.compute_node_centralities(graph)

    assert isinstance(centralities, dict)
    assert set(centralities.keys()) == set(graph.nodes())
    for node_metrics in centralities.values():
        assert CENTRALITY_KEYS.issubset(node_metrics.keys())


def test_get_top_nodes(sample_corr_df: pd.DataFrame, sample_sector_dict: dict[str, str]) -> None:
    graph = nw.build_network(sample_corr_df, threshold=0.3, sector_dict=sample_sector_dict)
    centralities = nw.compute_node_centralities(graph)

    top_nodes = nw.get_top_nodes(centralities, metric="pagerank", top_n=3)

    assert isinstance(top_nodes, list)
    assert len(top_nodes) <= 3
    assert all(isinstance(item, tuple) and len(item) == 2 for item in top_nodes)
    assert all(isinstance(item[0], str) for item in top_nodes)
    assert all(isinstance(item[1], float) for item in top_nodes)

    values = [value for _, value in top_nodes]
    assert values == sorted(values, reverse=True)


@pytest.mark.slow
def test_get_node_neighbors(real_graph_for_date: tuple[nx.Graph, pd.Timestamp]) -> None:
    graph, _ = real_graph_for_date
    neighbors = nw.get_node_neighbors(graph, "JPM")

    assert isinstance(neighbors, list)
    assert len(neighbors) > 0
    assert all(isinstance(n, tuple) and len(n) == 2 for n in neighbors)
    assert all(isinstance(name, str) for name, _ in neighbors)
    assert all(isinstance(weight, float) for _, weight in neighbors)

    abs_weights = [abs(weight) for _, weight in neighbors]
    assert abs_weights == sorted(abs_weights, reverse=True)
    assert neighbors[0][0] == "GS"


@pytest.mark.slow
def test_get_sector_subgraph(real_graph_for_date: tuple[nx.Graph, pd.Timestamp]) -> None:
    graph, _ = real_graph_for_date
    subgraph = nw.get_sector_subgraph(graph, "Financials")

    assert isinstance(subgraph, nx.Graph)
    assert subgraph.number_of_nodes() > 0
    assert all(data.get("sector") == "Financials" for _, data in subgraph.nodes(data=True))


def test_compare_networks(sample_corr_df: pd.DataFrame, sample_sector_dict: dict[str, str]) -> None:
    graph_before = nw.build_network(sample_corr_df, threshold=0.2, sector_dict=sample_sector_dict)
    graph_after = nw.build_network(sample_corr_df, threshold=0.5, sector_dict=sample_sector_dict)

    comparison = nw.compare_networks(graph_before, graph_after)

    assert isinstance(comparison, dict)
    assert len(comparison) > 0
    keys = set(comparison.keys())
    assert any(key.endswith("_before") for key in keys)
    assert any(key.endswith("_after") for key in keys)
    assert any(key.endswith("_delta") for key in keys)
