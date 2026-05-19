import sys
sys.path.insert(0, '.')

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.core import contagion as cg
from src.core import data_loader as dl
from src.core.network import build_network_for_date


SUMMARY_KEYS = {
    "shocked_node",
    "shock_magnitude",
    "model",
    "n_affected",
    "n_defaulted",
    "cascade_depth",
    "total_stress",
    "avg_stress",
    "top_10_affected",
}
MODEL_NAMES = ["linear_threshold", "debtrank", "cascade_removal"]


def _require_file(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"Missing required data file: {path}")


@pytest.fixture
def small_graph() -> nx.Graph:
    G = nx.Graph()
    for i in range(10):
        G.add_node(f"N{i}", ticker=f"N{i}", sector="TestSector")
    # Chain: N0-N1-N2-...-N9 with decreasing weights
    for i in range(9):
        w = 0.9 - i * 0.05
        G.add_edge(f"N{i}", f"N{i+1}", weight=w, abs_weight=w)
    # Add a hub: N0 connected to N5, N7
    G.add_edge("N0", "N5", weight=0.6, abs_weight=0.6)
    G.add_edge("N0", "N7", weight=0.5, abs_weight=0.5)
    return G


@pytest.fixture
def isolated_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_node("Iso", ticker="Iso", sector="TestSector")
    G.add_node("Other", ticker="Other", sector="TestSector")
    G.add_edge("Other", "Other2", weight=0.9, abs_weight=0.9)
    return G


@pytest.fixture(scope="module")
def real_graph() -> nx.Graph:
    _require_file(dl.NETWORKS / "correlation_matrices_pearson.pkl")
    _require_file(dl.FINAL / "sector_mapping.parquet")
    graph, _ = build_network_for_date("2025-12-01")
    return graph


def _assert_result_contract(result: cg.ShockResult, shocked_node: str) -> None:
    assert isinstance(result, cg.ShockResult)
    assert result.node_stress[shocked_node] >= 0.0
    assert result.n_affected >= 0
    assert result.cascade_depth >= 0
    assert all(0.0 <= v <= 1.0 for v in result.node_stress.values())

    summary = result.summary()
    assert isinstance(summary, dict)
    assert SUMMARY_KEYS.issubset(summary.keys())


def test_shockresult_properties() -> None:
    result = cg.ShockResult(
        shocked_node="N0",
        shock_magnitude=0.5,
        model="debtrank",
        node_stress={"N0": 0.5, "N1": 1.0, "N2": 0.2, "N3": 0.0},
        cascade_waves=[(1, ["N1"]), (2, ["N2"])],
    )

    assert result.n_affected == 2
    assert result.n_defaulted == 1
    assert result.cascade_depth == 2
    assert result.total_stress == pytest.approx(1.7)
    assert result.avg_stress == pytest.approx(0.425)
    assert result.affected_nodes[0][0] == "N1"
    assert result.affected_nodes[0][1] == pytest.approx(1.0)

    summary = result.summary()
    assert SUMMARY_KEYS.issubset(summary.keys())
    assert summary["shocked_node"] == "N0"
    assert summary["model"] == "debtrank"


@pytest.mark.parametrize(
    "model_func,kwargs",
    [
        (cg.linear_threshold, {"activation_threshold": 0.5}),
        (cg.debtrank, {}),
        (cg.cascade_removal, {"failure_threshold": 0.8}),
    ],
)
def test_models_return_valid_shock_result(small_graph: nx.Graph, model_func, kwargs: dict) -> None:
    result = model_func(small_graph, "N0", shock_magnitude=0.5, **kwargs)
    _assert_result_contract(result, "N0")
    assert result.node_stress["N0"] > 0.0


@pytest.mark.parametrize("model_func", [cg.linear_threshold, cg.debtrank, cg.cascade_removal])
def test_invalid_node_raises_value_error(small_graph: nx.Graph, model_func) -> None:
    with pytest.raises(ValueError):
        model_func(small_graph, "INVALID", shock_magnitude=0.5)


@pytest.mark.parametrize(
    "model_func,kwargs",
    [
        (cg.linear_threshold, {"activation_threshold": 0.5}),
        (cg.debtrank, {}),
        (cg.cascade_removal, {"failure_threshold": 0.8}),
    ],
)
def test_isolated_node_affects_only_itself(isolated_graph: nx.Graph, model_func, kwargs: dict) -> None:
    result = model_func(isolated_graph, "Iso", shock_magnitude=1.0, **kwargs)

    assert isinstance(result, cg.ShockResult)
    assert result.node_stress["Iso"] > 0.0
    assert result.n_affected == 0


@pytest.mark.parametrize(
    "model_func,kwargs",
    [
        (cg.linear_threshold, {"activation_threshold": 0.5}),
        (cg.debtrank, {}),
        (cg.cascade_removal, {"failure_threshold": 0.8}),
    ],
)
def test_shock_magnitude_zero_vs_one_propagation(small_graph: nx.Graph, model_func, kwargs: dict) -> None:
    low = model_func(small_graph, "N0", shock_magnitude=0.0, **kwargs)
    high = model_func(small_graph, "N0", shock_magnitude=1.0, **kwargs)

    assert isinstance(low, cg.ShockResult)
    assert isinstance(high, cg.ShockResult)
    assert low.total_stress <= high.total_stress
    assert low.n_affected <= high.n_affected
    assert all(0.0 <= v <= 1.0 for v in low.node_stress.values())
    assert all(0.0 <= v <= 1.0 for v in high.node_stress.values())


def test_run_shock_scenario(small_graph: nx.Graph) -> None:
    result = cg.run_shock_scenario(small_graph, shocked_node="N0", shock_magnitude=0.5, model="debtrank")

    assert isinstance(result, cg.ShockResult)
    assert result.model == "debtrank"
    assert result.node_stress["N0"] > 0.0


def test_compare_models(small_graph: nx.Graph) -> None:
    results = cg.compare_models(small_graph, shocked_node="N0", shock_magnitude=0.5)

    assert isinstance(results, dict)
    assert set(results.keys()) == set(MODEL_NAMES)
    assert all(isinstance(v, cg.ShockResult) for v in results.values())


@pytest.mark.slow
def test_debtrank_integration_real_graph(real_graph: nx.Graph) -> None:
    result = cg.debtrank(real_graph, shocked_node="JPM", shock_magnitude=0.5)

    assert isinstance(result, cg.ShockResult)
    assert result.n_affected > 50
