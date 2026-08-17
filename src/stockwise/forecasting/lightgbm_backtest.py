"""Leakage-safe recursive backtest for the fixed global LightGBM model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stockwise.forecasting.backtest import BASELINE_FOLDS
from stockwise.forecasting.features import build_leakage_safe_features, load_feature_input
from stockwise.metrics import mae, rmse, wape

TRAINING_WINDOW_DAYS = 730
HORIZON_DAYS = 28
CATEGORICAL_COLUMNS = ("item_id", "cat_id", "dept_id", "weekday")
MODEL_FEATURES = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "weekday",
    "wday",
    "month",
    "year",
    "is_event",
    "snap_CA",
    "item_id",
    "cat_id",
    "dept_id",
)
MODEL_PARAMETERS = {
    "objective": "regression_l1",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 4,
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "row_count": int(actual.size),
        "actual_units_sold": int(actual.sum()),
        "wape": float(wape(actual, predicted)),
        "mae": float(mae(actual, predicted)),
        "rmse": float(rmse(actual, predicted)),
    }


def _category_maps(sales: pd.DataFrame) -> dict[str, dict[object, int]]:
    return {
        column: {value: index for index, value in enumerate(sorted(sales[column].unique()))}
        for column in CATEGORICAL_COLUMNS
    }


def _model_matrix(features: pd.DataFrame, maps: dict[str, dict[object, int]]) -> pd.DataFrame:
    matrix = features.loc[:, MODEL_FEATURES].copy()
    for column, mapping in maps.items():
        matrix[column] = matrix[column].map(mapping).astype("int32")
    matrix["is_event"] = matrix["is_event"].astype("int8")
    return matrix


def recursive_forecast(
    model: object,
    history: pd.DataFrame,
    future_calendar: pd.DataFrame,
    maps: dict[str, dict[object, int]],
) -> np.ndarray:
    """Forecast one horizon recursively, intentionally ignoring future ``units_sold`` values."""
    item_ids = history["item_id"].drop_duplicates().sort_values().to_numpy()
    last_28 = history.loc[history["day_index"] >= history["day_index"].max() - 27]
    state = last_28.pivot(index="item_id", columns="day_index", values="units_sold").loc[item_ids]
    values = state.to_numpy(dtype="float32")
    predictions: list[np.ndarray] = []
    for day_index in sorted(future_calendar["day_index"].unique()):
        calendar = future_calendar.loc[future_calendar["day_index"] == day_index].set_index(
            "item_id"
        )
        frame = calendar.loc[item_ids].reset_index()
        frame["lag_1"] = values[:, -1]
        frame["lag_7"] = values[:, -7]
        frame["lag_14"] = values[:, -14]
        frame["lag_28"] = values[:, -28]
        frame["rolling_mean_7"] = values[:, -7:].mean(axis=1)
        frame["rolling_mean_28"] = values.mean(axis=1)
        frame["rolling_std_28"] = values.std(axis=1, ddof=1)
        predicted = np.clip(np.asarray(model.predict(_model_matrix(frame, maps))), 0, None).astype(
            "float32"
        )
        predictions.append(predicted)
        values = np.column_stack((values[:, 1:], predicted))
    return np.column_stack(predictions)


def _breakdown(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for fold, fold_frame in predictions.groupby("fold", sort=False):
        for value, group in fold_frame.groupby(column, sort=True):
            records.append(
                {
                    "fold": fold,
                    column: value,
                    **_metrics(group["units_sold"].to_numpy(), group["prediction"].to_numpy()),
                }
            )
    for value, group in predictions.groupby(column, sort=True):
        records.append(
            {
                "fold": "all_folds",
                column: value,
                **_metrics(group["units_sold"].to_numpy(), group["prediction"].to_numpy()),
            }
        )
    return pd.DataFrame(records)


def run_lightgbm_backtest(
    sales: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one fixed global model per outer fold and recursively forecast 28 future days."""
    from lightgbm import LGBMRegressor

    maps = _category_maps(sales)
    all_predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    for fold in BASELINE_FOLDS:
        train_start = fold.train_end_day - TRAINING_WINDOW_DAYS + 1
        context = sales.loc[
            (sales["day_index"] >= train_start - 28) & (sales["day_index"] <= fold.train_end_day)
        ]
        training = build_leakage_safe_features(context)
        training = training.loc[training["day_index"] >= train_start]
        model = LGBMRegressor(**MODEL_PARAMETERS)
        model.fit(
            _model_matrix(training, maps),
            training["units_sold"],
            categorical_feature=list(CATEGORICAL_COLUMNS),
        )
        history = sales.loc[
            (sales["day_index"] >= fold.train_end_day - 27)
            & (sales["day_index"] <= fold.train_end_day)
        ]
        future = sales.loc[
            (sales["day_index"] >= fold.validation_start_day)
            & (sales["day_index"] <= fold.validation_end_day)
        ]
        prediction_matrix = recursive_forecast(model, history, future, maps)
        ordered_future = future.sort_values(["item_id", "day_index"], kind="stable").copy()
        ordered_future["prediction"] = prediction_matrix.ravel()
        ordered_future["fold"] = fold.name
        all_predictions.append(ordered_future)
        fold_records.append(
            {
                "name": fold.name,
                "train_start_day": train_start,
                "train_end_day": fold.train_end_day,
                "validation_start_day": fold.validation_start_day,
                "validation_end_day": fold.validation_end_day,
                **_metrics(
                    ordered_future["units_sold"].to_numpy(), ordered_future["prediction"].to_numpy()
                ),
            }
        )
        importances.append(
            pd.DataFrame({"feature": MODEL_FEATURES, "importance": model.feature_importances_})
        )
    predictions = pd.concat(all_predictions, ignore_index=True)
    combined = _metrics(predictions["units_sold"].to_numpy(), predictions["prediction"].to_numpy())
    fold_metrics = pd.DataFrame([*fold_records, {"name": "all_folds", **combined}])
    importance = (
        pd.concat(importances)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False, kind="stable")
    )
    summary = {
        "model": "lightgbm_global",
        "training_window_days": TRAINING_WINDOW_DAYS,
        "horizon_days": HORIZON_DAYS,
        "folds": fold_records,
        "all_folds_combined": combined,
        "parameters": MODEL_PARAMETERS,
    }
    return (
        summary,
        fold_metrics,
        _breakdown(predictions, "cat_id"),
        _breakdown(predictions, "dept_id"),
        predictions,
        importance,
    )


def write_lightgbm_artifacts(
    processed_directory: str | Path, output_directory: str | Path
) -> dict[str, object]:
    """Run the fixed global-model backtest and write only evaluation artifacts."""
    sales = load_feature_input(processed_directory)
    summary, folds, categories, departments, predictions, importance = run_lightgbm_backtest(sales)
    baseline_path = Path("artifacts/baseline/summary.json")
    if baseline_path.is_file():
        baseline_wape = json.loads(baseline_path.read_text())["all_folds_combined"]["wape"]
        summary["baseline_comparison"] = {
            "baseline_wape": baseline_wape,
            "wape_improvement_percentage_points": (
                baseline_wape - summary["all_folds_combined"]["wape"]
            )
            * 100,
            "wape_relative_improvement_percent": (
                baseline_wape - summary["all_folds_combined"]["wape"]
            )
            / baseline_wape
            * 100,
        }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "model_configuration.json").write_text(json.dumps(MODEL_PARAMETERS, indent=2) + "\n")
    folds.to_csv(output / "fold_metrics.csv", index=False)
    categories.to_csv(output / "metrics_by_category.csv", index=False)
    departments.to_csv(output / "metrics_by_department.csv", index=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    return summary
