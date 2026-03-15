"""Lightweight systemic-risk forecasting helpers for fixed-origin backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BASE_FEATURES = ["density", "avg_abs_weight", "avg_clustering", "regime_numeric"]
LAGS = [0, 1, 5, 20]
REGIME_BUCKETS = ("calm", "stress")
DIRECT_TARGETS = ["density", "avg_abs_weight", "avg_clustering", "regime_numeric"]


@dataclass
class RidgeModel:
    feature_names: list[str]
    mean_: pd.Series
    scale_: pd.Series
    coef_: np.ndarray
    intercept_: float
    alpha: float = 1.0

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        X = frame[self.feature_names].astype(float)
        X_std = (X - self.mean_) / self.scale_
        preds = X_std.to_numpy() @ self.coef_ + self.intercept_
        return pd.Series(preds, index=frame.index, dtype=float)


@dataclass
class RegimeAwareRidgeModel:
    global_model: RidgeModel
    bucket_models: dict[str, RidgeModel]

    def predict(self, frame: pd.DataFrame, regime_value: float) -> pd.Series:
        bucket = regime_bucket(regime_value)
        model = self.bucket_models.get(bucket, self.global_model)
        return model.predict(frame)


@dataclass
class DirectForecastBundle:
    feature_names: list[str]
    models_by_horizon: dict[int, dict[str, RidgeModel]]

    def available_horizons(self) -> list[int]:
        return sorted(self.models_by_horizon.keys())


def build_forecast_frame(network_metrics: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    merged = network_metrics[["density", "avg_abs_weight", "avg_clustering"]].join(
        regime_data[["Regime_Numeric"]],
        how="inner",
    )
    merged = merged.rename(columns={"Regime_Numeric": "regime_numeric"}).sort_index()
    merged["regime_numeric"] = merged["regime_numeric"].astype(float)
    merged["risk_pressure"] = (
        merged["density"] * merged["avg_abs_weight"] * (1.0 + merged["regime_numeric"])
    ).astype(float)

    for base in BASE_FEATURES:
        for lag in LAGS:
            if lag == 0:
                merged[f"{base}_lag0"] = merged[base].astype(float)
            else:
                merged[f"{base}_lag{lag}"] = merged[base].shift(lag).astype(float)
    return merged.dropna().copy()


def build_direct_feature_frame(network_features: pd.DataFrame, regime_data: pd.DataFrame) -> pd.DataFrame:
    frame = network_features.copy().sort_index()
    aligned_regime = regime_data[["Regime_Numeric"]].rename(columns={"Regime_Numeric": "regime_numeric"})
    frame = frame.join(aligned_regime, how="inner")
    if "vix" not in frame.columns and "VIX" in regime_data.columns:
        frame = frame.join(regime_data[["VIX"]].rename(columns={"VIX": "vix"}), how="left")
    if "density" not in frame.columns:
        frame["density"] = 0.0
    if "avg_clustering" not in frame.columns:
        frame["avg_clustering"] = 0.0
    if "avg_abs_weight" not in frame.columns:
        if "abs_weight" in frame.columns:
            frame["avg_abs_weight"] = pd.to_numeric(frame["abs_weight"], errors="coerce").abs()
        elif "avg_weight" in frame.columns:
            frame["avg_abs_weight"] = pd.to_numeric(frame["avg_weight"], errors="coerce").abs()
        else:
            frame["avg_abs_weight"] = 0.0
    if "regime_numeric" not in frame.columns:
        frame["regime_numeric"] = 0.0
    frame["regime_numeric"] = frame["regime_numeric"].astype(float)
    frame["risk_pressure"] = (
        frame["density"].astype(float) * frame["avg_abs_weight"].astype(float) * (1.0 + frame["regime_numeric"])
    ).astype(float)
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    frame[numeric_cols] = frame[numeric_cols].ffill().bfill()
    ordered_cols = [c for c in numeric_cols if c not in {"risk_pressure"}] + ["risk_pressure"]
    return frame[ordered_cols].dropna().copy()


def _feature_names() -> list[str]:
    return [f"{base}_lag{lag}" for base in BASE_FEATURES for lag in LAGS]


def fit_ridge(
    frame: pd.DataFrame,
    target_col: str,
    alpha: float = 1.0,
    feature_names: list[str] | None = None,
) -> RidgeModel:
    feature_names = feature_names or _feature_names()
    X = frame[feature_names].astype(float)
    y = frame[target_col].astype(float)

    mean_ = X.mean()
    scale_ = X.std(ddof=0).replace(0.0, 1.0)
    X_std = ((X - mean_) / scale_).to_numpy()
    y_np = y.to_numpy()

    xtx = X_std.T @ X_std
    ridge = xtx + alpha * np.eye(xtx.shape[0])
    coef_ = np.linalg.solve(ridge, X_std.T @ y_np)
    # Predictions are made on standardized features directly, so the fitted
    # intercept lives in standardized space and should equal the target mean.
    intercept_ = float(y_np.mean())
    return RidgeModel(
        feature_names=feature_names,
        mean_=mean_,
        scale_=scale_,
        coef_=coef_,
        intercept_=intercept_,
        alpha=alpha,
    )


def fit_recursive_models(frame: pd.DataFrame, alpha: float = 1.0) -> dict[str, RidgeModel]:
    train = frame.copy()
    models: dict[str, RidgeModel] = {}
    for target in BASE_FEATURES:
        labeled = train.copy()
        labeled[target] = labeled[target].shift(-1)
        labeled = labeled.dropna()
        models[target] = fit_ridge(labeled, target, alpha=alpha)
    return models


def regime_bucket(value: float) -> str:
    return "stress" if float(value) >= 2.0 else "calm"


def fit_regime_aware_recursive_models(
    frame: pd.DataFrame,
    alpha: float = 1.0,
    min_bucket_samples: int = 40,
) -> dict[str, RegimeAwareRidgeModel]:
    train = frame.copy()
    models: dict[str, RegimeAwareRidgeModel] = {}
    for target in BASE_FEATURES:
        labeled = train.copy()
        labeled[target] = labeled[target].shift(-1)
        labeled = labeled.dropna()
        labeled["regime_bucket"] = labeled["regime_numeric_lag0"].map(regime_bucket)
        global_model = fit_ridge(labeled, target, alpha=alpha)
        bucket_models: dict[str, RidgeModel] = {}
        for bucket in REGIME_BUCKETS:
            bucket_frame = labeled[labeled["regime_bucket"] == bucket].copy()
            if len(bucket_frame) < min_bucket_samples:
                continue
            bucket_models[bucket] = fit_ridge(bucket_frame, target, alpha=alpha)
        models[target] = RegimeAwareRidgeModel(global_model=global_model, bucket_models=bucket_models)
    return models


def fit_direct_multi_horizon_models(
    frame: pd.DataFrame,
    horizons: list[int],
    alpha: float = 1.0,
) -> DirectForecastBundle:
    feature_names = [c for c in frame.columns if c not in set(DIRECT_TARGETS + ["risk_pressure"])]
    models_by_horizon: dict[int, dict[str, RidgeModel]] = {}
    base = frame.copy()
    for horizon in sorted(set(int(h) for h in horizons if int(h) > 0)):
        labeled = base.copy()
        horizon_models: dict[str, RidgeModel] = {}
        for target in DIRECT_TARGETS:
            future = labeled[target].shift(-horizon)
            if target == "regime_numeric":
                labeled_target = future
            else:
                labeled_target = future - labeled[target]
            train = labeled[feature_names].copy()
            train[target] = labeled_target
            train = train.dropna()
            horizon_models[target] = fit_ridge(train, target, alpha=alpha, feature_names=feature_names)
        models_by_horizon[horizon] = horizon_models
    return DirectForecastBundle(feature_names=feature_names, models_by_horizon=models_by_horizon)


def _state_to_frame(state: dict[str, list[float]]) -> pd.DataFrame:
    row: dict[str, float] = {}
    for base in BASE_FEATURES:
        history = state[base]
        for lag in LAGS:
            row[f"{base}_lag{lag}"] = float(history[-(lag + 1)])
    return pd.DataFrame([row])


def recursive_forecast(
    frame: pd.DataFrame,
    models: dict[str, RidgeModel],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = frame.sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = list(frame.index)
    anchor_dates = [d for d in all_dates if d <= start_ts]
    if not anchor_dates:
        raise ValueError("No training history available before start_date.")
    anchor = anchor_dates[-1]

    state = {
        base: frame.loc[:anchor, base].tail(max(LAGS) + 1).astype(float).tolist()
        for base in BASE_FEATURES
    }
    if any(len(values) < max(LAGS) + 1 for values in state.values()):
        raise ValueError("Insufficient history for recursive forecast.")

    target_dates = [d for d in all_dates if start_ts < d <= end_ts]
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date_ts in target_dates:
        features = _state_to_frame(state)
        next_state: dict[str, float] = {}
        for base in BASE_FEATURES:
            pred = float(models[base].predict(features).iloc[0])
            if base == "regime_numeric":
                pred = float(np.clip(np.round(pred), 0, 4))
            else:
                pred = float(max(pred, 0.0))
            next_state[base] = pred
            state[base].append(pred)
            state[base] = state[base][-(max(LAGS) + 1) :]
        rows.append({"date": date_ts, **next_state})

    forecast = pd.DataFrame(rows).set_index("date").sort_index()
    forecast["risk_pressure"] = (
        forecast["density"] * forecast["avg_abs_weight"] * (1.0 + forecast["regime_numeric"])
    ).astype(float)
    return forecast


def recursive_regime_aware_forecast(
    frame: pd.DataFrame,
    models: dict[str, RegimeAwareRidgeModel],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = frame.sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = list(frame.index)
    anchor_dates = [d for d in all_dates if d <= start_ts]
    if not anchor_dates:
        raise ValueError("No training history available before start_date.")
    anchor = anchor_dates[-1]

    state = {
        base: frame.loc[:anchor, base].tail(max(LAGS) + 1).astype(float).tolist()
        for base in BASE_FEATURES
    }
    if any(len(values) < max(LAGS) + 1 for values in state.values()):
        raise ValueError("Insufficient history for recursive forecast.")

    target_dates = [d for d in all_dates if start_ts < d <= end_ts]
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date_ts in target_dates:
        features = _state_to_frame(state)
        active_regime = float(state["regime_numeric"][-1])
        next_state: dict[str, float] = {}
        for base in BASE_FEATURES:
            pred = float(models[base].predict(features, regime_value=active_regime).iloc[0])
            if base == "regime_numeric":
                pred = float(np.clip(np.round(pred), 0, 4))
            else:
                pred = float(max(pred, 0.0))
            next_state[base] = pred
            state[base].append(pred)
            state[base] = state[base][-(max(LAGS) + 1) :]
        rows.append({"date": date_ts, **next_state})

    forecast = pd.DataFrame(rows).set_index("date").sort_index()
    forecast["risk_pressure"] = (
        forecast["density"] * forecast["avg_abs_weight"] * (1.0 + forecast["regime_numeric"])
    ).astype(float)
    return forecast


def direct_multi_horizon_forecast(
    feature_frame: pd.DataFrame,
    bundle: DirectForecastBundle,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    feature_frame = feature_frame.sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = list(feature_frame.index)
    anchor_dates = [d for d in all_dates if d <= start_ts]
    if not anchor_dates:
        raise ValueError("No training history available before start_date.")
    anchor = anchor_dates[-1]

    target_dates = [d for d in all_dates if start_ts < d <= end_ts]
    if not target_dates:
        raise ValueError("No target dates available for direct forecast.")

    anchor_row = feature_frame.loc[[anchor], bundle.feature_names].astype(float)
    anchor_levels = feature_frame.loc[anchor, DIRECT_TARGETS].astype(float).to_dict()
    max_available = max(bundle.available_horizons())
    rows: list[dict[str, float | pd.Timestamp]] = []
    for horizon, date_ts in enumerate(target_dates, start=1):
        use_h = min(horizon, max_available)
        horizon_models = bundle.models_by_horizon[use_h]
        pred_row: dict[str, float | pd.Timestamp] = {"date": date_ts}
        for target in DIRECT_TARGETS:
            pred = float(horizon_models[target].predict(anchor_row).iloc[0])
            if not np.isfinite(pred):
                pred = anchor_levels[target]
            if target == "regime_numeric":
                pred_row[target] = float(np.clip(np.round(pred), 0, 4))
            else:
                pred_row[target] = float(max(anchor_levels[target] + pred, 0.0))
        rows.append(pred_row)

    forecast = pd.DataFrame(rows).set_index("date").sort_index()
    forecast["risk_pressure"] = (
        forecast["density"] * forecast["avg_abs_weight"] * (1.0 + forecast["regime_numeric"])
    ).astype(float)
    return forecast


def evaluate_regime(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return {
            "accuracy": 0.0,
            "high_or_worse_recall": None,
            "crisis_recall": None,
        }
    actual_i = aligned["actual"].astype(int)
    pred_i = aligned["predicted"].astype(int)
    accuracy = float((actual_i == pred_i).mean()) if len(actual_i) else 0.0
    high_mask = actual_i >= 3
    high_recall = float(((pred_i[high_mask] >= 3).mean())) if high_mask.any() else None
    crisis_mask = actual_i == 4
    crisis_recall = float(((pred_i[crisis_mask] == 4).mean())) if crisis_mask.any() else None
    return {
        "accuracy": accuracy,
        "high_or_worse_recall": high_recall,
        "crisis_recall": crisis_recall,
    }


def evaluate_scalar(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return {"mae": 0.0, "rmse": 0.0}
    err = aligned["predicted"] - aligned["actual"]
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
    }


def top_k_overlap_forecast(
    node_centralities: dict[pd.Timestamp, dict[str, dict[str, float]]],
    anchor_date: str,
    start_date: str,
    end_date: str,
    metric: str = "pagerank",
    k: int = 5,
) -> dict[str, float]:
    anchor_ts = max(d for d in node_centralities if d <= pd.Timestamp(anchor_date))
    predicted = [
        ticker
        for ticker, _ in sorted(
            node_centralities[anchor_ts].items(),
            key=lambda item: item[1].get(metric, 0.0),
            reverse=True,
        )[:k]
    ]

    overlaps: list[float] = []
    for date_ts in sorted(d for d in node_centralities if pd.Timestamp(start_date) < d <= pd.Timestamp(end_date)):
        actual = [
            ticker
            for ticker, _ in sorted(
                node_centralities[date_ts].items(),
                key=lambda item: item[1].get(metric, 0.0),
                reverse=True,
            )[:k]
        ]
        overlap = len(set(predicted) & set(actual)) / float(k)
        overlaps.append(overlap)

    return {
        "top_k": k,
        "anchor_date": str(anchor_ts.date()),
        "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "min_overlap": float(np.min(overlaps)) if overlaps else 0.0,
        "max_overlap": float(np.max(overlaps)) if overlaps else 0.0,
    }


def fit_regime_transition(regime_numeric: pd.Series) -> dict[int, int]:
    states = sorted({int(v) for v in regime_numeric.dropna().unique()})
    transition_map: dict[int, int] = {}
    series = regime_numeric.astype(int)
    for state in states:
        nxt = series.shift(-1)[series == state].dropna().astype(int)
        if nxt.empty:
            transition_map[state] = state
        else:
            transition_map[state] = int(nxt.value_counts().idxmax())
    return transition_map


def recursive_baseline_forecast(
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = frame.sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_dates = list(frame.index)
    anchor_dates = [d for d in all_dates if d <= start_ts]
    if not anchor_dates:
        raise ValueError("No training history available before start_date.")
    anchor = anchor_dates[-1]
    transition = fit_regime_transition(frame.loc[:anchor, "regime_numeric"])

    target_dates = [d for d in all_dates if start_ts < d <= end_ts]
    current = {
        "density": float(frame.loc[anchor, "density"]),
        "avg_abs_weight": float(frame.loc[anchor, "avg_abs_weight"]),
        "avg_clustering": float(frame.loc[anchor, "avg_clustering"]),
        "regime_numeric": float(frame.loc[anchor, "regime_numeric"]),
    }
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date_ts in target_dates:
        current_regime = int(current["regime_numeric"])
        next_regime = float(transition.get(current_regime, current_regime))
        rows.append(
            {
                "date": date_ts,
                "density": float(current["density"]),
                "avg_abs_weight": float(current["avg_abs_weight"]),
                "avg_clustering": float(current["avg_clustering"]),
                "regime_numeric": next_regime,
            }
        )
        current["regime_numeric"] = next_regime

    forecast = pd.DataFrame(rows).set_index("date").sort_index()
    forecast["risk_pressure"] = (
        forecast["density"] * forecast["avg_abs_weight"] * (1.0 + forecast["regime_numeric"])
    ).astype(float)
    return forecast


def nearest_leq(index: pd.Index, target: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(target)
    candidates = [pd.Timestamp(v) for v in index if pd.Timestamp(v) <= ts]
    if not candidates:
        raise ValueError(f"No index value <= {ts}.")
    return candidates[-1]


def nearest_geq(index: pd.Index, target: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(target)
    candidates = [pd.Timestamp(v) for v in index if pd.Timestamp(v) >= ts]
    if not candidates:
        raise ValueError(f"No index value >= {ts}.")
    return candidates[0]


def build_walk_forward_folds(
    index: pd.Index,
    eval_start: str,
    eval_end: str,
    step_days: int = 20,
    horizon_days: int = 20,
    min_train_points: int = 120,
) -> list[dict[str, str]]:
    ordered = [pd.Timestamp(v) for v in index]
    start_ts = nearest_geq(index, eval_start)
    end_ts = nearest_leq(index, eval_end)
    folds: list[dict[str, str]] = []

    start_pos = ordered.index(start_ts)
    end_pos = ordered.index(end_ts)
    origin_positions = range(start_pos, end_pos, step_days)
    for pos in origin_positions:
        if pos < min_train_points:
            continue
        test_end_pos = min(pos + horizon_days, end_pos)
        if test_end_pos <= pos:
            continue
        train_end = ordered[pos]
        test_end = ordered[test_end_pos]
        folds.append(
            {
                "train_end": str(train_end.date()),
                "test_end": str(test_end.date()),
            }
        )
    return folds


def aggregate_fold_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        return {"n_folds": 0}

    def _avg(path1: str, path2: str | None = None) -> float | None:
        values: list[float] = []
        for rep in reports:
            cur: object = rep
            for key in path1.split("."):
                cur = cur[key]  # type: ignore[index]
            if path2 is not None and cur is not None:
                cur = cur[path2]  # type: ignore[index]
            if cur is None:
                continue
            values.append(float(cur))
        if not values:
            return None
        return float(np.mean(values))

    selected = [str(rep.get("best_model", rep.get("model", "unknown"))) for rep in reports]
    return {
        "n_folds": len(reports),
        "best_model_counts": pd.Series(selected).value_counts().to_dict(),
        "regime_accuracy_mean": _avg("regime", "accuracy"),
        "density_mae_mean": _avg("density", "mae"),
        "avg_abs_weight_mae_mean": _avg("avg_abs_weight", "mae"),
        "avg_clustering_mae_mean": _avg("avg_clustering", "mae"),
        "risk_pressure_mae_mean": _avg("risk_pressure", "mae"),
        "top_k_overlap_mean": _avg("top_systemic_nodes_persistence", "mean_overlap"),
    }


def evaluate_joined(joined: pd.DataFrame, model_name: str, train_end_ts: pd.Timestamp) -> dict[str, object]:
    report: dict[str, object] = {
        "model": model_name,
        "train_end": str(train_end_ts.date()),
        "test_start": str(joined.index.min().date()),
        "test_end": str(joined.index.max().date()),
        "n_test_days": int(len(joined)),
        "regime": evaluate_regime(joined["regime_numeric"], joined["pred_regime_numeric"]),
        "density": evaluate_scalar(joined["density"], joined["pred_density"]),
        "avg_abs_weight": evaluate_scalar(joined["avg_abs_weight"], joined["pred_avg_abs_weight"]),
        "avg_clustering": evaluate_scalar(joined["avg_clustering"], joined["pred_avg_clustering"]),
        "risk_pressure": evaluate_scalar(joined["risk_pressure"], joined["pred_risk_pressure"]),
        "examples": [],
    }
    sample_dates = [joined.index[0], joined.index[min(len(joined) // 2, len(joined) - 1)], joined.index[-1]]

    def _safe_int(value: object) -> int | None:
        try:
            val = float(value)
        except Exception:
            return None
        if not np.isfinite(val):
            return None
        return int(val)

    def _safe_float(value: object) -> float | None:
        try:
            val = float(value)
        except Exception:
            return None
        if not np.isfinite(val):
            return None
        return val

    for date_ts in sample_dates:
        row = joined.loc[date_ts]
        report["examples"].append(
            {
                "date": str(date_ts.date()),
                "actual_regime_numeric": _safe_int(row["regime_numeric"]),
                "pred_regime_numeric": _safe_int(row["pred_regime_numeric"]),
                "actual_density": _safe_float(row["density"]),
                "pred_density": _safe_float(row["pred_density"]),
                "actual_avg_abs_weight": _safe_float(row["avg_abs_weight"]),
                "pred_avg_abs_weight": _safe_float(row["pred_avg_abs_weight"]),
            }
        )
    return report


def run_backtest_on_frame(
    frame: pd.DataFrame,
    node_centralities: dict[pd.Timestamp, dict[str, dict[str, float]]],
    direct_feature_frame: pd.DataFrame | None,
    train_end: str,
    test_end: str,
    alpha: float,
    include_direct_model: bool = True,
) -> tuple[dict[str, object], pd.DataFrame]:
    reports_by_model, joineds_by_model, best_name = run_backtest_models_on_frame(
        frame,
        direct_feature_frame=direct_feature_frame,
        train_end=train_end,
        test_end=test_end,
        alpha=alpha,
        include_direct_model=include_direct_model,
    )
    ridge_report = reports_by_model["ridge_recursive"]
    regime_aware_report = reports_by_model["ridge_regime_aware"]
    baseline_report = reports_by_model["persistence_markov"]
    best_report = reports_by_model[best_name]
    best_joined = joineds_by_model[best_name]
    train_end_ts = nearest_leq(frame.index, train_end)
    test_end_ts = nearest_leq(frame.index, test_end)
    final_report = dict(best_report)
    final_report["best_model"] = best_name
    alternatives: dict[str, object] = {
        "ridge_recursive": ridge_report if best_name != "ridge_recursive" else {"selected": True},
        "ridge_regime_aware": regime_aware_report if best_name != "ridge_regime_aware" else {"selected": True},
        "persistence_markov": baseline_report if best_name != "persistence_markov" else {"selected": True},
    }
    if include_direct_model and "direct_rich_ridge" in reports_by_model:
        direct_report = reports_by_model["direct_rich_ridge"]
        alternatives["direct_rich_ridge"] = (
            direct_report if best_name != "direct_rich_ridge" else {"selected": True}
        )
    final_report["alternatives"] = alternatives
    final_report["top_systemic_nodes_persistence"] = top_k_overlap_forecast(
        node_centralities,
        anchor_date=str(train_end_ts.date()),
        start_date=str(train_end_ts.date()),
        end_date=str(test_end_ts.date()),
        metric="pagerank",
        k=5,
    )
    return final_report, best_joined


def run_backtest_models_on_frame(
    frame: pd.DataFrame,
    direct_feature_frame: pd.DataFrame | None,
    train_end: str,
    test_end: str,
    alpha: float,
    include_direct_model: bool = True,
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame], str]:
    train_end_ts = nearest_leq(frame.index, train_end)
    test_end_ts = nearest_leq(frame.index, test_end)

    train_frame = frame.loc[:train_end_ts].copy()
    actual = frame.loc[(frame.index > train_end_ts) & (frame.index <= test_end_ts)].copy()
    if actual.empty:
        raise ValueError(f"Empty holdout window for train_end={train_end} test_end={test_end}.")

    models = fit_recursive_models(train_frame, alpha=alpha)
    regime_aware_models = fit_regime_aware_recursive_models(train_frame, alpha=alpha)
    ridge_forecast = recursive_forecast(
        frame,
        models=models,
        start_date=str(train_end_ts.date()),
        end_date=str(test_end_ts.date()),
    )
    regime_aware_forecast = recursive_regime_aware_forecast(
        frame,
        models=regime_aware_models,
        start_date=str(train_end_ts.date()),
        end_date=str(test_end_ts.date()),
    )
    baseline_forecast = recursive_baseline_forecast(
        frame,
        start_date=str(train_end_ts.date()),
        end_date=str(test_end_ts.date()),
    )
    if include_direct_model and direct_feature_frame is not None:
        direct_frame = direct_feature_frame.loc[:test_end_ts].copy()
        direct_target_dates = [d for d in direct_frame.index if train_end_ts < d <= test_end_ts]
        if direct_target_dates:
            horizons = list(range(1, len(direct_target_dates) + 1))
            direct_bundle = fit_direct_multi_horizon_models(
                direct_frame.loc[:train_end_ts].copy(),
                horizons=horizons,
                alpha=alpha,
            )
            direct_forecast = direct_multi_horizon_forecast(
                direct_frame,
                bundle=direct_bundle,
                start_date=str(train_end_ts.date()),
                end_date=str(test_end_ts.date()),
            )
        else:
            direct_forecast = baseline_forecast.copy()
    else:
        direct_forecast = baseline_forecast.copy()

    ridge_joined = actual.join(ridge_forecast.add_prefix("pred_"), how="inner")
    regime_aware_joined = actual.join(regime_aware_forecast.add_prefix("pred_"), how="inner")
    baseline_joined = actual.join(baseline_forecast.add_prefix("pred_"), how="inner")

    ridge_report = evaluate_joined(ridge_joined, model_name="ridge_recursive", train_end_ts=train_end_ts)
    regime_aware_report = evaluate_joined(
        regime_aware_joined,
        model_name="ridge_regime_aware",
        train_end_ts=train_end_ts,
    )
    baseline_report = evaluate_joined(baseline_joined, model_name="persistence_markov", train_end_ts=train_end_ts)
    reports = {
        "ridge_recursive": ridge_report,
        "ridge_regime_aware": regime_aware_report,
        "persistence_markov": baseline_report,
    }
    joineds = {
        "ridge_recursive": ridge_joined,
        "ridge_regime_aware": regime_aware_joined,
        "persistence_markov": baseline_joined,
    }
    if include_direct_model:
        direct_joined = actual.join(direct_forecast.add_prefix("pred_"), how="inner")
        direct_report = evaluate_joined(
            direct_joined,
            model_name="direct_rich_ridge",
            train_end_ts=train_end_ts,
        )
        reports["direct_rich_ridge"] = direct_report
        joineds["direct_rich_ridge"] = direct_joined

    def _score(rep: dict[str, object]) -> tuple[float, float]:
        regime_acc = float(rep["regime"]["accuracy"])  # type: ignore[index]
        density_mae = float(rep["density"]["mae"])  # type: ignore[index]
        return regime_acc, -density_mae

    best_name = max(reports, key=lambda name: _score(reports[name]))
    return (
        reports,
        joineds,
        best_name,
    )


def run_full_evaluation_on_frame(
    frame: pd.DataFrame,
    node_centralities: dict[pd.Timestamp, dict[str, dict[str, float]]],
    direct_feature_frame: pd.DataFrame | None,
    train_end: str,
    test_end: str,
    alpha: float,
    walk_step_days: int,
    walk_horizon_days: int,
    include_direct_model: bool = True,
) -> tuple[dict[str, object], pd.DataFrame]:
    fixed_report, fixed_joined = run_backtest_on_frame(
        frame,
        node_centralities,
        direct_feature_frame=direct_feature_frame,
        train_end=train_end,
        test_end=test_end,
        alpha=alpha,
        include_direct_model=include_direct_model,
    )

    latest_eval_end = nearest_leq(frame.index, test_end)
    one_year_before = latest_eval_end - pd.DateOffset(years=1)
    eval_start = nearest_geq(frame.index, one_year_before)
    walk_folds = build_walk_forward_folds(
        frame.index,
        eval_start=str(eval_start.date()),
        eval_end=str(latest_eval_end.date()),
        step_days=walk_step_days,
        horizon_days=walk_horizon_days,
        min_train_points=max(120, walk_horizon_days + 40),
    )
    walk_reports: list[dict[str, object]] = []
    for fold in walk_folds:
        rep, _ = run_backtest_on_frame(
            frame,
            node_centralities,
            direct_feature_frame=direct_feature_frame,
            train_end=fold["train_end"],
            test_end=fold["test_end"],
            alpha=alpha,
            include_direct_model=include_direct_model,
        )
        walk_reports.append(rep)

    stress_windows = [
        ("COVID-19 Crash", "2020-02-20", "2020-04-07"),
        ("Russia-Ukraine 2022", "2022-02-24", "2022-03-15"),
        ("SVB Crisis 2023", "2023-03-08", "2023-03-24"),
        ("Japan Carry Trade 2024", "2024-08-01", "2024-08-15"),
    ]
    stress_reports: list[dict[str, object]] = []
    ordered = [pd.Timestamp(v) for v in frame.index]
    first_idx = pd.Timestamp(frame.index.min())
    for label, start_date, end_date in stress_windows:
        start_ts = nearest_geq(frame.index, start_date)
        end_ts = nearest_leq(frame.index, end_date)
        start_pos = ordered.index(start_ts)
        if start_pos <= 120 or start_ts <= first_idx:
            continue
        train_end_ts = ordered[start_pos - 1]
        rep, _ = run_backtest_on_frame(
            frame,
            node_centralities,
            direct_feature_frame=direct_feature_frame,
            train_end=str(train_end_ts.date()),
            test_end=str(end_ts.date()),
            alpha=alpha,
            include_direct_model=include_direct_model,
        )
        rep = dict(rep)
        rep["fold_label"] = label
        stress_reports.append(rep)

    return (
        {
            "fixed_origin": fixed_report,
            "walk_forward_last_year": {
                "eval_start": str(eval_start.date()),
                "eval_end": str(latest_eval_end.date()),
                "step_days": walk_step_days,
                "horizon_days": walk_horizon_days,
                "summary": aggregate_fold_reports(walk_reports),
                "folds": walk_reports,
            },
            "historical_stress_folds": {
                "summary": aggregate_fold_reports(stress_reports),
                "folds": stress_reports,
            },
        },
        fixed_joined,
    )
