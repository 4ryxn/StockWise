"""Fixed rolling-origin evaluation for StockWise forecasting baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stockwise.forecasting.baseline import seasonal_naive_forecast
from stockwise.metrics import mae, rmse, wape

BACKTEST_COLUMNS = ["item_id", "cat_id", "dept_id", "day_index", "units_sold"]


@dataclass(frozen=True)
class BaselineFold:
    """Inclusive day-index boundaries for one expanding-window validation fold."""

    name: str
    train_start_day: int
    train_end_day: int
    validation_start_day: int
    validation_end_day: int


BASELINE_FOLDS = (
    BaselineFold("fold_1", 1, 1857, 1858, 1885),
    BaselineFold("fold_2", 1, 1885, 1886, 1913),
    BaselineFold("fold_3", 1, 1913, 1914, 1941),
)


def load_backtest_sales(processed_directory: str | Path) -> pd.DataFrame:
    """Load the columns needed for baseline evaluation from every partition."""
    processed_path = Path(processed_directory)
    partitions = sorted(processed_path.glob("part-*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {processed_path}")
    return pd.concat(
        [pd.read_parquet(partition, columns=BACKTEST_COLUMNS) for partition in partitions],
        ignore_index=True,
    )


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    return {
        "row_count": int(actual.size),
        "actual_units_sold": int(actual.sum()),
        "wape": float(wape(actual, predicted)) if actual.sum() else None,
        "mae": float(mae(actual, predicted)),
        "rmse": float(rmse(actual, predicted)),
    }


def _forecast_fold(
    demand: pd.DataFrame,
    fold: BaselineFold,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected_horizon = fold.validation_end_day - fold.validation_start_day + 1
    if expected_horizon != horizon:
        raise ValueError(f"{fold.name} validation window must equal the {horizon}-day horizon")

    demand_matrix = demand.pivot(index="item_id", columns="day_index", values="units_sold")
    expected_days = set(range(fold.train_start_day, fold.validation_end_day + 1))
    missing_days = sorted(expected_days - set(demand_matrix.columns))
    if missing_days:
        raise ValueError(f"{fold.name} is missing day indexes: {missing_days[:5]}")
    history_days = list(range(fold.train_end_day - 6, fold.train_end_day + 1))
    validation_days = list(range(fold.validation_start_day, fold.validation_end_day + 1))
    history = demand_matrix.loc[:, history_days].to_numpy(dtype=float)
    actual = demand_matrix.loc[:, validation_days].to_numpy(dtype=float)
    predicted = np.vstack(
        [seasonal_naive_forecast(item_history, horizon=horizon) for item_history in history]
    )
    return actual, np.clip(predicted, a_min=0, a_max=None)


def _breakdowns(
    demand: pd.DataFrame,
    fold_results: list[tuple[BaselineFold, np.ndarray, np.ndarray]],
    group_column: str,
) -> pd.DataFrame:
    item_groups = demand.drop_duplicates("item_id").set_index("item_id")[group_column]
    item_ids = demand["item_id"].drop_duplicates().sort_values().tolist()
    groups = item_groups.reindex(item_ids)
    records: list[dict[str, object]] = []

    for fold, actual, predicted in fold_results:
        for group in sorted(groups.dropna().unique()):
            metrics = _metric_values(actual[groups.eq(group)], predicted[groups.eq(group)])
            records.append({"fold": fold.name, group_column: group, **metrics})

    for group in sorted(groups.dropna().unique()):
        actual = np.concatenate([actual[groups.eq(group)] for _, actual, _ in fold_results])
        predicted = np.concatenate(
            [predicted[groups.eq(group)] for _, _, predicted in fold_results]
        )
        metrics = _metric_values(actual, predicted)
        records.append({"fold": "all_folds", group_column: group, **metrics})
    return pd.DataFrame(records)


def backtest_seasonal_naive(
    demand: pd.DataFrame,
    folds: tuple[BaselineFold, ...] = BASELINE_FOLDS,
    horizon: int = 28,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a clipped weekly seasonal-naive forecast over fixed rolling folds."""
    missing_columns = sorted(set(BACKTEST_COLUMNS) - set(demand.columns))
    if missing_columns:
        raise ValueError(f"Demand data is missing required columns: {', '.join(missing_columns)}")
    if demand.duplicated(["item_id", "day_index"]).any():
        raise ValueError("Demand data contains duplicate item-day rows")

    ordered_demand = demand.sort_values(["item_id", "day_index"], kind="stable")
    fold_results: list[tuple[BaselineFold, np.ndarray, np.ndarray]] = []
    fold_records: list[dict[str, object]] = []
    for fold in folds:
        actual, predicted = _forecast_fold(ordered_demand, fold, horizon)
        fold_results.append((fold, actual, predicted))
        fold_records.append({**asdict(fold), **_metric_values(actual, predicted)})

    combined_actual = np.concatenate([actual.ravel() for _, actual, _ in fold_results])
    combined_predicted = np.concatenate([predicted.ravel() for _, _, predicted in fold_results])
    combined_metrics = _metric_values(combined_actual, combined_predicted)
    summary = {
        "model": "seasonal_naive",
        "season_length_days": 7,
        "horizon_days": horizon,
        "folds": fold_records,
        "all_folds_combined": combined_metrics,
    }
    combined_fold_record = {
        "name": "all_folds",
        "train_start_day": None,
        "train_end_day": None,
        "validation_start_day": None,
        "validation_end_day": None,
        **combined_metrics,
    }
    return (
        summary,
        pd.DataFrame([*fold_records, combined_fold_record]),
        _breakdowns(ordered_demand, fold_results, "cat_id"),
        _breakdowns(ordered_demand, fold_results, "dept_id"),
    )


def write_baseline_artifacts(
    processed_directory: str | Path,
    output_directory: str | Path,
    folds: tuple[BaselineFold, ...] = BASELINE_FOLDS,
    horizon: int = 28,
) -> dict[str, object]:
    """Run the fixed baseline backtest and write its readable summary and breakdowns."""
    summary, fold_metrics, category_metrics, department_metrics = backtest_seasonal_naive(
        load_backtest_sales(processed_directory), folds=folds, horizon=horizon
    )
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    fold_metrics.to_csv(output_path / "fold_metrics.csv", index=False)
    category_metrics.to_csv(output_path / "metrics_by_category.csv", index=False)
    department_metrics.to_csv(output_path / "metrics_by_department.csv", index=False)
    return summary
