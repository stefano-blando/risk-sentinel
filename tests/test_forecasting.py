import pandas as pd

from src.core.forecasting import (
    aggregate_fold_reports,
    build_direct_feature_frame,
    build_walk_forward_folds,
    build_forecast_frame,
    evaluate_regime,
    fit_recursive_models,
    fit_regime_aware_recursive_models,
    fit_ridge,
    recursive_forecast,
    recursive_regime_aware_forecast,
)


def test_recursive_forecast_produces_holdout_path() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80)
    network_metrics = pd.DataFrame(
        {
            "density": [0.20 + i * 0.001 for i in range(80)],
            "avg_abs_weight": [0.35 + i * 0.0005 for i in range(80)],
            "avg_clustering": [0.45 + i * 0.0004 for i in range(80)],
        },
        index=dates,
    )
    regime_data = pd.DataFrame(
        {
            "Regime_Numeric": [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20,
        },
        index=dates,
    )

    frame = build_forecast_frame(network_metrics, regime_data)
    train_end = str(frame.index[30].date())
    test_end = str(frame.index[-1].date())
    models = fit_recursive_models(frame.loc[:train_end])
    forecast = recursive_forecast(frame, models=models, start_date=train_end, end_date=test_end)

    assert not forecast.empty
    assert forecast.index.min() > pd.Timestamp(train_end)
    assert forecast.index.max() == pd.Timestamp(test_end)
    assert {"density", "avg_abs_weight", "avg_clustering", "regime_numeric", "risk_pressure"}.issubset(
        forecast.columns
    )
    assert forecast["regime_numeric"].between(0, 4).all()


def test_regime_aware_recursive_forecast_produces_holdout_path() -> None:
    dates = pd.bdate_range("2025-01-02", periods=120)
    network_metrics = pd.DataFrame(
        {
            "density": [0.20 + i * 0.0015 for i in range(120)],
            "avg_abs_weight": [0.35 + i * 0.0007 for i in range(120)],
            "avg_clustering": [0.45 + i * 0.0006 for i in range(120)],
        },
        index=dates,
    )
    regime_data = pd.DataFrame(
        {
            "Regime_Numeric": [0] * 40 + [1] * 30 + [2] * 25 + [3] * 25,
        },
        index=dates,
    )

    frame = build_forecast_frame(network_metrics, regime_data)
    train_end = str(frame.index[70].date())
    test_end = str(frame.index[-1].date())
    models = fit_regime_aware_recursive_models(frame.loc[:train_end], alpha=1.0, min_bucket_samples=10)
    forecast = recursive_regime_aware_forecast(frame, models=models, start_date=train_end, end_date=test_end)

    assert not forecast.empty
    assert forecast.index.min() > pd.Timestamp(train_end)
    assert forecast.index.max() == pd.Timestamp(test_end)
    assert forecast["regime_numeric"].between(0, 4).all()


def test_walk_forward_folds_and_aggregate_reports() -> None:
    dates = pd.bdate_range("2025-01-02", periods=260)
    folds = build_walk_forward_folds(
        dates,
        eval_start="2025-09-01",
        eval_end="2025-12-31",
        step_days=20,
        horizon_days=20,
        min_train_points=120,
    )
    assert folds
    assert folds[0]["train_end"] < folds[0]["test_end"]

    summary = aggregate_fold_reports(
        [
            {
                "best_model": "persistence_markov",
                "regime": {"accuracy": 0.5},
                "density": {"mae": 0.01},
                "avg_abs_weight": {"mae": 0.02},
                "avg_clustering": {"mae": 0.03},
                "risk_pressure": {"mae": 0.04},
                "top_systemic_nodes_persistence": {"mean_overlap": 0.4},
            },
            {
                "best_model": "ridge_recursive",
                "regime": {"accuracy": 0.3},
                "density": {"mae": 0.05},
                "avg_abs_weight": {"mae": 0.06},
                "avg_clustering": {"mae": 0.07},
                "risk_pressure": {"mae": 0.08},
                "top_systemic_nodes_persistence": {"mean_overlap": 0.2},
            },
        ]
    )
    assert summary["n_folds"] == 2
    assert summary["best_model_counts"]["persistence_markov"] == 1
    assert summary["regime_accuracy_mean"] == 0.4


def test_fit_ridge_predicts_training_mean_on_standardized_features() -> None:
    frame = pd.DataFrame(
        {
            "density_lag0": [0.20, 0.25, 0.30, 0.35],
            "density_lag1": [0.19, 0.20, 0.25, 0.30],
            "density_lag5": [0.18, 0.18, 0.19, 0.20],
            "density_lag20": [0.17, 0.17, 0.18, 0.18],
            "avg_abs_weight_lag0": [0.40, 0.41, 0.42, 0.43],
            "avg_abs_weight_lag1": [0.39, 0.40, 0.41, 0.42],
            "avg_abs_weight_lag5": [0.38, 0.38, 0.39, 0.40],
            "avg_abs_weight_lag20": [0.37, 0.37, 0.38, 0.38],
            "avg_clustering_lag0": [0.50, 0.51, 0.52, 0.53],
            "avg_clustering_lag1": [0.49, 0.50, 0.51, 0.52],
            "avg_clustering_lag5": [0.48, 0.48, 0.49, 0.50],
            "avg_clustering_lag20": [0.47, 0.47, 0.48, 0.48],
            "regime_numeric_lag0": [0.0, 1.0, 1.0, 2.0],
            "regime_numeric_lag1": [0.0, 0.0, 1.0, 1.0],
            "regime_numeric_lag5": [0.0, 0.0, 0.0, 1.0],
            "regime_numeric_lag20": [0.0, 0.0, 0.0, 0.0],
            "target": [0.30, 0.35, 0.40, 0.45],
        }
    )
    model = fit_ridge(frame, "target", alpha=1.0)
    pred = model.predict(frame.iloc[[0]]).iloc[0]

    assert pred > 0.0
    assert abs(pred - frame["target"].mean()) < 0.2


def test_evaluate_regime_ignores_non_finite_values() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    predicted = pd.Series([1.0, float("nan"), float("inf")])
    metrics = evaluate_regime(actual, predicted)

    assert metrics["accuracy"] == 1.0


def test_build_direct_feature_frame_handles_missing_avg_abs_weight() -> None:
    dates = pd.bdate_range("2025-01-02", periods=5)
    network_features = pd.DataFrame(
        {
            "density": [0.20, 0.21, 0.22, 0.23, 0.24],
            "avg_weight": [-0.30, -0.25, -0.20, -0.15, -0.10],
            "avg_clustering": [0.40, 0.41, 0.42, 0.43, 0.44],
        },
        index=dates,
    )
    regime_data = pd.DataFrame(
        {
            "Regime_Numeric": [0, 1, 1, 2, 2],
            "VIX": [15.0, 17.0, 18.0, 21.0, 22.0],
        },
        index=dates,
    )

    frame = build_direct_feature_frame(network_features, regime_data)

    assert "avg_abs_weight" in frame.columns
    assert (frame["avg_abs_weight"] >= 0.0).all()
    assert "risk_pressure" in frame.columns
    assert frame["risk_pressure"].notna().all()
