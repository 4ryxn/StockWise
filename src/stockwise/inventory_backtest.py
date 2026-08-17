"""Scenario inventory simulation; M5 sales are not observed inventory records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_SCENARIO = {
    "lead_time_days": 7,
    "review_period_days": 7,
    "service_level": 0.95,
    "demand_assumption": "lost_sales",
    "is_simulation": True,
    "note": (
        "Scenario simulation only: M5 provides sales, not observed inventory, "
        "orders, or lost sales."
    ),
}


def _order_up_to(
    mean: float, std: float, scenario: dict[str, object], safety_stock_multiplier: float = 1.0
) -> float:
    protection = int(scenario["lead_time_days"]) + int(scenario["review_period_days"])
    z_score = NormalDist().inv_cdf(float(scenario["service_level"]))
    return mean * protection + safety_stock_multiplier * z_score * std * math.sqrt(protection)


def simulate_item_policy(
    actual: np.ndarray,
    predictions: np.ndarray,
    pre_fold_history: np.ndarray,
    policy: str,
    scenario: dict[str, object] = DEFAULT_SCENARIO,
    safety_stock_multiplier: float = 1.0,
) -> dict[str, float | int | list[float]]:
    """Simulate one item with orders based only on prior observations and predictions."""
    lead_time, review = int(scenario["lead_time_days"]), int(scenario["review_period_days"])
    history = list(np.asarray(pre_fold_history, dtype=float)[-28:])
    initial_mean, initial_std = float(np.mean(history)), float(np.std(history, ddof=1))
    on_hand = _order_up_to(initial_mean, initial_std, scenario, safety_stock_multiplier)
    incoming = np.zeros(len(actual) + lead_time, dtype=float)
    fulfilled = stockout_units = total_order = on_hand_sum = 0.0
    stockout_days = 0
    orders: list[float] = []
    for day, demand in enumerate(actual.astype(float)):
        on_hand += incoming[day]
        if day % review == 0:
            observed = np.asarray(history[-28:], dtype=float)
            historical_mean, historical_std = float(observed.mean()), float(observed.std(ddof=1))
            if policy == "fixed_historical":
                target = _order_up_to(
                    historical_mean, historical_std, scenario, safety_stock_multiplier
                )
            elif policy == "stockwise":
                remaining_predictions = predictions[day : day + lead_time + review]
                padded = np.pad(
                    remaining_predictions,
                    (0, lead_time + review - len(remaining_predictions)),
                    constant_values=historical_mean,
                )
                target = (
                    float(padded.sum())
                    + NormalDist().inv_cdf(float(scenario["service_level"]))
                    * historical_std
                    * math.sqrt(lead_time + review)
                    * safety_stock_multiplier
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")
            order = max(0.0, target - (on_hand + incoming[day + 1 :].sum()))
            incoming[day + lead_time] += order
            total_order += order
            orders.append(order)
        sold = min(on_hand, demand)
        lost = demand - sold
        on_hand -= sold
        fulfilled += sold
        stockout_units += lost
        stockout_days += int(lost > 0)
        on_hand_sum += on_hand
        history.append(demand)
    demand_total = float(actual.sum())
    return {
        "demand_units": demand_total,
        "fulfilled_units": fulfilled,
        "stockout_units": stockout_units,
        "stockout_days": stockout_days,
        "average_on_hand_units": on_hand_sum / len(actual),
        "ending_inventory_units": on_hand,
        "total_ordered_units": total_order,
        "fill_rate": fulfilled / demand_total if demand_total else 1.0,
        "orders": orders,
    }


def _aggregate(items: pd.DataFrame) -> dict[str, float | int]:
    return {
        "demand_units": float(items["demand_units"].sum()),
        "stockout_units": float(items["stockout_units"].sum()),
        "stockout_days": int(items["stockout_days"].sum()),
        "average_on_hand_units": float(items["average_on_hand_units"].mean()),
        "total_ordered_units": float(items["total_ordered_units"].sum()),
        "fill_rate": float(items["fulfilled_units"].sum() / items["demand_units"].sum()),
    }


def run_inventory_backtest(
    history: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Simulate both policies for every prediction fold and item."""
    records: list[dict[str, object]] = []
    for fold, fold_predictions in predictions.groupby("fold", sort=True):
        start = int(fold_predictions["day_index"].min())
        historical = history.loc[
            (history["day_index"] >= start - 28) & (history["day_index"] < start)
        ]
        histories = {
            item: group["units_sold"].to_numpy() for item, group in historical.groupby("item_id")
        }
        for item_id, group in fold_predictions.groupby("item_id", sort=False):
            ordered = group.sort_values("day_index")
            for policy in ("fixed_historical", "stockwise"):
                metrics = simulate_item_policy(
                    ordered["units_sold"].to_numpy(),
                    ordered["prediction"].to_numpy(),
                    histories[item_id],
                    policy,
                )
                records.append(
                    {
                        "fold": fold,
                        "item_id": item_id,
                        "policy": policy,
                        **{key: value for key, value in metrics.items() if key != "orders"},
                    }
                )
    item_metrics = pd.DataFrame(records)
    fold_metrics = pd.DataFrame(
        [
            {"fold": fold, "policy": policy, **_aggregate(group)}
            for (fold, policy), group in item_metrics.groupby(["fold", "policy"])
        ]
    )
    comparison = pd.DataFrame(
        [
            {"policy": policy, **_aggregate(group)}
            for policy, group in item_metrics.groupby("policy")
        ]
    )
    summary = {
        "scenario_simulation": True,
        "policies": comparison.to_dict(orient="records"),
        "folds": fold_metrics.to_dict(orient="records"),
    }
    return summary, comparison, fold_metrics, item_metrics


def write_inventory_artifacts(
    processed_directory: str | Path, prediction_path: str | Path, output_directory: str | Path
) -> dict[str, object]:
    """Run and persist the inventory scenario simulation without changing source data."""
    partitions = sorted(Path(processed_directory).glob("part-*.parquet"))
    history = pd.concat(
        [
            pd.read_parquet(part, columns=["item_id", "day_index", "units_sold"])
            for part in partitions
        ],
        ignore_index=True,
    )
    summary, comparison, folds, items = run_inventory_backtest(
        history, pd.read_parquet(prediction_path)
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "scenario.json").write_text(json.dumps(DEFAULT_SCENARIO, indent=2) + "\n")
    comparison.to_csv(output / "policy_comparison.csv", index=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    items.to_parquet(output / "item_fold_metrics.parquet", index=False)
    return summary
