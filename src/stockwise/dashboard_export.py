"""Compact static-data exports for the frontend dashboard and planner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

PLANNER_FOLD = "fold_3"
PLANNER_ITEM_COUNT_PER_CATEGORY = 10


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def _select_planner_items(
    fold_predictions: pd.DataFrame,
    historical_demand: pd.DataFrame,
    *,
    items_per_category: int = PLANNER_ITEM_COUNT_PER_CATEGORY,
) -> dict:
    """Build a deterministic, small item-level planner payload.

    The sample only uses recursive fold-3 predictions for selection. It takes the highest
    positive 28-day forecast volume in each category, with ``item_id`` as a stable tie-breaker.
    Historical standard deviation only uses rows before the fold-3 start day.
    """
    required_prediction_columns = {
        "item_id",
        "cat_id",
        "dept_id",
        "day_index",
        "prediction",
    }
    missing = required_prediction_columns.difference(fold_predictions.columns)
    if missing:
        raise ValueError(f"Planner predictions are missing columns: {sorted(missing)}")
    if historical_demand.empty:
        raise ValueError("Planner historical demand is empty")

    fold_start_day = int(fold_predictions["day_index"].min())
    horizon_days = int(fold_predictions["day_index"].nunique())
    totals = (
        fold_predictions.groupby(["item_id", "cat_id", "dept_id"], as_index=False)
        .agg(forecast_total=("prediction", "sum"), forecast_days=("day_index", "nunique"))
        .query("forecast_total > 0 and forecast_days == @horizon_days")
        .sort_values(["cat_id", "forecast_total", "item_id"], ascending=[True, False, True])
    )
    selected = totals.groupby("cat_id", group_keys=False).head(items_per_category).copy()
    expected_count = items_per_category * int(totals["cat_id"].nunique())
    if len(selected) != expected_count:
        raise ValueError("Not enough positive complete-horizon items for the planner sample")

    history = historical_demand.loc[
        historical_demand["day_index"] < fold_start_day, ["item_id", "units_sold"]
    ]
    standard_deviations = (
        history.groupby("item_id", as_index=False)["units_sold"].std(ddof=0).fillna(0.0)
    ).rename(columns={"units_sold": "historical_daily_demand_std"})
    selected = selected.merge(standard_deviations, on="item_id", how="left")
    if selected["historical_daily_demand_std"].isna().any():
        raise ValueError("Planner items are missing pre-fold historical demand")

    item_rows: list[dict] = []
    sort_columns = ["cat_id", "forecast_total", "item_id"]
    for item in selected.sort_values(sort_columns, ascending=[True, False, True]).itertuples(
        index=False
    ):
        item_forecast = fold_predictions.loc[
            fold_predictions["item_id"] == item.item_id, ["day_index", "prediction"]
        ].sort_values("day_index")
        item_rows.append(
            {
                "item_id": item.item_id,
                "category": item.cat_id,
                "department": item.dept_id,
                "forecast": [round(float(value), 4) for value in item_forecast["prediction"]],
                "historical_daily_demand_std": round(
                    float(item.historical_daily_demand_std), 4
                ),
            }
        )

    return {
        "forecast_fold": PLANNER_FOLD,
        "forecast_start_day": fold_start_day,
        "forecast_horizon_days": horizon_days,
        "selection_method": (
            "Deterministic category-balanced sample: for each CA_1 category, select the "
            f"{items_per_category} items with the highest positive total recursive {PLANNER_FOLD} "
            "forecast across its complete 28-day horizon; item_id breaks ties."
        ),
        "historical_variability_method": (
            "Population standard deviation of daily units sold using only days before the "
            "fold-3 forecast start."
        ),
        "items": item_rows,
    }


def write_planner_data(
    prediction_path: str | Path,
    processed_directory: str | Path,
    output_directory: str | Path,
) -> dict:
    """Write a compact planner payload without exposing source or full prediction data."""
    predictions = pd.read_parquet(prediction_path, filters=[("fold", "==", PLANNER_FOLD)])
    selected_ids = (
        predictions.groupby(["item_id", "cat_id", "dept_id"], as_index=False)["prediction"]
        .sum()
        .query("prediction > 0")
        .sort_values(["cat_id", "prediction", "item_id"], ascending=[True, False, True])
        .groupby("cat_id", group_keys=False)
        .head(PLANNER_ITEM_COUNT_PER_CATEGORY)["item_id"]
        .tolist()
    )
    parquet_files = sorted(Path(processed_directory).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet partitions found in {processed_directory}")
    historical = pd.read_parquet(
        parquet_files,
        columns=["item_id", "day_index", "units_sold"],
        filters=[("item_id", "in", selected_ids)],
    )
    data = _select_planner_items(predictions, historical)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "planner.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def write_dashboard_data(
    artifacts_directory: str | Path,
    output_directory: str | Path,
    processed_directory: str | Path = "data/processed/ca_1",
) -> dict:
    """Write static dashboard evidence and a compact item-level planner payload."""
    artifacts = Path(artifacts_directory)

    def load(path: Path) -> dict:
        return json.loads(path.read_text())

    eda = load(artifacts / "eda/summary.json")
    lightgbm = load(artifacts / "lightgbm/summary.json")
    inventory = load(artifacts / "inventory/summary.json")
    sensitivity = load(artifacts / "inventory_sensitivity/summary.json")
    data = {
        "overview": {
            "row_count": eda["row_count"],
            "item_count": eda["item_count"],
            "wape": lightgbm["all_folds_combined"]["wape"],
            "improvement": lightgbm["baseline_comparison"]["wape_relative_improvement_percent"],
        },
        "folds": lightgbm["folds"],
        "categories": rows(artifacts / "lightgbm/metrics_by_category.csv"),
        "departments": rows(artifacts / "lightgbm/metrics_by_department.csv"),
        "importance": rows(artifacts / "lightgbm/feature_importance.csv"),
        "inventory": inventory["policies"],
        "sensitivity": rows(artifacts / "inventory_sensitivity/scenario_metrics.csv"),
        "pareto": rows(artifacts / "inventory_sensitivity/pareto_frontier.csv"),
        "methodology": [
            "28-day rolling held-out backtests across M5 CA_1.",
            "LightGBM uses lag, rolling, ID, and known calendar features only.",
            "Forecasts are backtest results, not live predictions.",
            "Inventory is a lost-sales scenario simulation; M5 has no observed inventory.",
        ],
        "note": sensitivity["note"],
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dashboard.json").write_text(json.dumps(data, indent=2) + "\n")
    write_planner_data(artifacts / "lightgbm/predictions.parquet", processed_directory, output)
    return data
