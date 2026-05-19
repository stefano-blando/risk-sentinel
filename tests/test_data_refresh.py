import pandas as pd

from src.core.data_refresh import (
    build_network_features,
    build_regime_frame,
    classify_regime,
    compute_incremental_network_outputs,
)


def test_classify_regime_uses_expected_buckets() -> None:
    vix = pd.Series([12.0, 18.0, 22.0, 28.0, 40.0])
    labels = classify_regime(vix)
    assert labels.tolist() == ["Calm", "Normal", "Elevated", "High", "Crisis"]


def test_build_regime_frame_and_incremental_snapshots() -> None:
    dates = pd.bdate_range("2026-01-02", periods=8)
    returns = pd.DataFrame(
        {
            "AAA": [0.00, 0.01, 0.02, -0.01, 0.01, 0.00, 0.01, -0.01],
            "BBB": [0.00, 0.02, 0.01, -0.01, 0.00, 0.01, 0.01, -0.02],
            "CCC": [0.00, -0.01, 0.00, 0.02, 0.01, 0.00, -0.01, 0.01],
            "DDD": [0.00, 0.00, 0.01, 0.01, -0.01, 0.02, 0.00, 0.01],
        },
        index=dates,
    )
    returns.index.name = "Date"

    close_prices = (1.0 + returns).cumprod() * 100.0
    sp500 = pd.Series([5000, 5010, 5030, 5005, 5020, 5035, 5040, 5050], index=dates)
    vix = pd.Series([14.0, 15.5, 17.0, 18.0, 21.0, 24.0, 26.0, 30.0], index=dates)
    market_data, regime_data = build_regime_frame(returns, sp500, vix)

    existing_date = dates[2]
    existing_corr = {existing_date: returns.iloc[:3].corr().fillna(0.0)}
    existing_node_centralities = {
        existing_date: {
            ticker: {
                "degree": 0.0,
                "betweenness": 0.0,
                "closeness": 0.0,
                "eigenvector": 0.0,
                "pagerank": 0.0,
            }
            for ticker in returns.columns
        }
    }
    sector_dict = {"AAA": "Financials", "BBB": "Financials", "CCC": "Energy", "DDD": "Energy"}

    all_corr, all_node_centralities, feature_core, sector_features, metrics = compute_incremental_network_outputs(
        returns_simple=returns,
        market_data=market_data,
        regime_data=regime_data,
        sector_dict=sector_dict,
        existing_corr_matrices=existing_corr,
        existing_node_centralities=existing_node_centralities,
        threshold=0.1,
        window=3,
    )

    assert max(all_corr) == dates[-1]
    assert max(all_node_centralities) == dates[-1]
    assert not feature_core.empty
    assert not sector_features.empty
    assert not metrics.empty
    assert {"density", "avg_degree", "vix", "modularity"}.issubset(feature_core.columns)
    assert "sect_financials_degree" in sector_features.columns
    assert {"nodes", "edges", "density", "avg_degree", "regime"}.issubset(metrics.columns)

    network_features = build_network_features(feature_core, sector_features)
    assert "density_delta_1d" in network_features.columns
    assert "density_pct_5d" in network_features.columns
    assert "systemic_pressure" in network_features.columns
    assert network_features.index.max() == dates[-1]
    assert close_prices.shape[0] == len(dates)
