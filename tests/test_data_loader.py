import sys
sys.path.insert(0, '.')

from pathlib import Path

import pandas as pd
import pytest

from src.core import data_loader as dl


REQUIRED_REGIMES = {"Calm", "Normal", "Elevated", "High", "Crisis"}
CENTRALITY_COLUMNS = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]


def _require_file(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"Missing required data file: {path}")


def _assert_non_empty_df(df: pd.DataFrame) -> None:
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def _assert_has_columns(df: pd.DataFrame, expected: list[str]) -> None:
    missing = [col for col in expected if col not in df.columns]
    assert not missing, f"Missing expected columns: {missing}"


def test_load_sector_mapping() -> None:
    _require_file(dl.FINAL / "sector_mapping.parquet")
    sector_mapping = dl.load_sector_mapping()

    _assert_non_empty_df(sector_mapping)
    assert sector_mapping.shape[0] == 210
    _assert_has_columns(sector_mapping, ["Ticker", "Sector"])
    assert sector_mapping["Sector"].nunique() == 11


def test_get_sector_dict() -> None:
    _require_file(dl.FINAL / "sector_mapping.parquet")
    sector_dict = dl.get_sector_dict()

    assert isinstance(sector_dict, dict)
    assert len(sector_dict) == 210
    assert all(isinstance(k, str) for k in sector_dict.keys())
    assert all(isinstance(v, str) for v in sector_dict.values())


def test_get_ticker_list() -> None:
    _require_file(dl.FINAL / "sector_mapping.parquet")
    tickers = dl.get_ticker_list()

    assert isinstance(tickers, list)
    assert len(tickers) == 210
    assert all(isinstance(t, str) for t in tickers)


def test_load_close_prices() -> None:
    _require_file(dl.FINAL / "close_prices.parquet")
    _require_file(dl.FINAL / "sector_mapping.parquet")

    close_prices = dl.load_close_prices()
    tickers = set(dl.get_ticker_list())

    _assert_non_empty_df(close_prices)
    # Price table should include ticker columns.
    assert tickers.issubset(set(close_prices.columns))


def test_load_returns() -> None:
    _require_file(dl.FINAL / "returns_simple.parquet")
    _require_file(dl.FINAL / "sector_mapping.parquet")

    returns_df = dl.load_returns()
    tickers = set(dl.get_ticker_list())

    _assert_non_empty_df(returns_df)
    assert tickers.issubset(set(returns_df.columns))


def test_load_market_data() -> None:
    _require_file(dl.FINAL / "market_data.parquet")
    market = dl.load_market_data()

    _assert_non_empty_df(market)
    _assert_has_columns(market, ["VIX", "SP500", "SP500_Return"])


def test_load_regime_data() -> None:
    _require_file(dl.FINAL / "regime_data.parquet")
    regimes = dl.load_regime_data()

    _assert_non_empty_df(regimes)
    _assert_has_columns(regimes, ["Regime"])
    assert set(regimes["Regime"].dropna().unique()).issubset(REQUIRED_REGIMES)


def test_load_network_metrics() -> None:
    _require_file(dl.FINAL / "network_metrics.parquet")
    network_metrics = dl.load_network_metrics()

    _assert_non_empty_df(network_metrics)
    _assert_has_columns(
        network_metrics,
        [
            "nodes",
            "edges",
            "density",
            "avg_degree",
            "avg_clustering",
            "n_components",
            "largest_cc",
            "largest_cc_pct",
            "avg_weight",
            "avg_abs_weight",
            "method",
            "vix",
            "regime",
        ],
    )


def test_load_network_features() -> None:
    _require_file(dl.FINAL / "network_features.parquet")
    network_features = dl.load_network_features()

    _assert_non_empty_df(network_features)
    _assert_has_columns(network_features, ["density", "avg_degree"])


def test_load_sector_centralities() -> None:
    _require_file(dl.FINAL / "sector_centrality_features.parquet")
    sector_centralities = dl.load_sector_centralities()

    _assert_non_empty_df(sector_centralities)
    assert any("degree" in str(col).lower() for col in sector_centralities.columns)


def test_load_node_centralities() -> None:
    _require_file(dl.NETWORKS / "node_centralities.pkl")
    centralities = dl.load_node_centralities()

    assert isinstance(centralities, dict)
    assert len(centralities) > 0

    first_date = next(iter(centralities))
    assert isinstance(first_date, pd.Timestamp)

    first_date_values = centralities[first_date]
    assert isinstance(first_date_values, dict)
    assert len(first_date_values) > 0

    first_ticker_values = next(iter(first_date_values.values()))
    assert isinstance(first_ticker_values, dict)
    assert set(CENTRALITY_COLUMNS).issubset(set(first_ticker_values.keys()))


def test_get_available_dates() -> None:
    _require_file(dl.NETWORKS / "node_centralities.pkl")
    dates = dl.get_available_dates()

    assert isinstance(dates, list)
    assert len(dates) > 0
    assert all(isinstance(d, pd.Timestamp) for d in dates)
    assert dates == sorted(dates)


def test_find_nearest_date_exact_and_approximate() -> None:
    dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-10"),
        pd.Timestamp("2020-01-20"),
    ]

    assert dl.find_nearest_date("2020-01-10", dates) == pd.Timestamp("2020-01-10")
    assert dl.find_nearest_date("2020-01-12", dates) == pd.Timestamp("2020-01-10")


@pytest.mark.slow
def test_get_correlation_matrix() -> None:
    _require_file(dl.NETWORKS / "correlation_matrices_pearson.pkl")
    _require_file(dl.NETWORKS / "node_centralities.pkl")

    available_dates = dl.get_available_dates()
    assert available_dates, "No available dates found for correlation matrix test"

    requested = str(available_dates[0].date())
    corr, actual_date = dl.get_correlation_matrix(requested)

    _assert_non_empty_df(corr)
    assert isinstance(actual_date, pd.Timestamp)
    assert corr.shape[0] == corr.shape[1]


def test_get_node_centralities_for_date() -> None:
    _require_file(dl.NETWORKS / "node_centralities.pkl")
    available_dates = dl.get_available_dates()
    assert available_dates, "No available dates found for node centralities test"

    requested = str(available_dates[0].date())
    node_data, actual_date = dl.get_node_centralities_for_date(requested)

    assert isinstance(node_data, dict)
    assert len(node_data) > 0
    assert isinstance(actual_date, pd.Timestamp)


def test_centralities_to_dataframe() -> None:
    _require_file(dl.NETWORKS / "node_centralities.pkl")
    centralities = dl.load_node_centralities()
    sample_date = next(iter(centralities))
    node_centralities = centralities[sample_date]

    df = dl.centralities_to_dataframe(node_centralities)

    _assert_non_empty_df(df)
    assert list(df.columns) == CENTRALITY_COLUMNS


def test_load_mvp_data() -> None:
    _require_file(dl.FINAL / "sector_mapping.parquet")
    _require_file(dl.FINAL / "market_data.parquet")
    _require_file(dl.FINAL / "regime_data.parquet")
    _require_file(dl.FINAL / "network_metrics.parquet")
    _require_file(dl.FINAL / "network_features.parquet")
    _require_file(dl.NETWORKS / "node_centralities.pkl")

    mvp = dl.load_mvp_data()

    assert isinstance(mvp, dict)
    assert set(mvp.keys()) == {
        "sector_mapping",
        "sector_dict",
        "tickers",
        "market",
        "regimes",
        "network_metrics",
        "network_features",
        "node_centralities",
    }

    _assert_non_empty_df(mvp["sector_mapping"])
    assert isinstance(mvp["sector_dict"], dict)
    assert isinstance(mvp["tickers"], list)
    _assert_non_empty_df(mvp["market"])
    _assert_non_empty_df(mvp["regimes"])
    _assert_non_empty_df(mvp["network_metrics"])
    _assert_non_empty_df(mvp["network_features"])
    assert isinstance(mvp["node_centralities"], dict)
