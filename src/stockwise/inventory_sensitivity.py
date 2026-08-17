"""Scenario exploration for forecast-driven inventory safety-stock choices."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockwise.inventory_backtest import DEFAULT_SCENARIO, _aggregate, simulate_item_policy

SERVICE_LEVELS = (0.90, 0.95, 0.975, 0.99)
SAFETY_STOCK_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5, 2.0)


def run_inventory_sensitivity(history: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Run fixed scenario combinations without selecting a winning outer-fold configuration."""
    records: list[dict[str, object]] = []
    for service_level in SERVICE_LEVELS:
        for multiplier in SAFETY_STOCK_MULTIPLIERS:
            item_records: list[dict[str, object]] = []
            scenario = {**DEFAULT_SCENARIO, "service_level": service_level}
            for _fold, fold_predictions in predictions.groupby("fold", sort=True):
                start = int(fold_predictions["day_index"].min())
                pre_fold = history.loc[
                    (history["day_index"] >= start - 28) & (history["day_index"] < start)
                ]
                histories = {
                    item: group["units_sold"].to_numpy()
                    for item, group in pre_fold.groupby("item_id")
                }
                for item_id, group in fold_predictions.groupby("item_id", sort=False):
                    ordered = group.sort_values("day_index")
                    metrics = simulate_item_policy(
                        ordered["units_sold"].to_numpy(),
                        ordered["prediction"].to_numpy(),
                        histories[item_id],
                        "stockwise",
                        scenario,
                        multiplier,
                    )
                    item_records.append(
                        {key: value for key, value in metrics.items() if key != "orders"}
                    )
            records.append(
                {
                    "service_level": service_level,
                    "safety_stock_multiplier": multiplier,
                    **_aggregate(pd.DataFrame(item_records)),
                }
            )
    return pd.DataFrame(records)


def pareto_frontier(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep choices not dominated on higher fill rate and lower average on-hand inventory."""
    frontier = []
    for _, candidate in metrics.iterrows():
        dominates = (metrics["fill_rate"] >= candidate["fill_rate"]) & (
            metrics["average_on_hand_units"] <= candidate["average_on_hand_units"]
        )
        strictly_better = (metrics["fill_rate"] > candidate["fill_rate"]) | (
            metrics["average_on_hand_units"] < candidate["average_on_hand_units"]
        )
        if not (dominates & strictly_better).any():
            frontier.append(candidate)
    return pd.DataFrame(frontier).sort_values("average_on_hand_units").reset_index(drop=True)


def write_inventory_sensitivity_artifacts(
    processed_directory: str | Path, prediction_path: str | Path, output_directory: str | Path
) -> dict[str, object]:
    """Write scenario-exploration artifacts without changing baseline inventory results."""
    partitions = sorted(Path(processed_directory).glob("part-*.parquet"))
    history = pd.concat(
        [
            pd.read_parquet(part, columns=["item_id", "day_index", "units_sold"])
            for part in partitions
        ],
        ignore_index=True,
    )
    metrics = run_inventory_sensitivity(history, pd.read_parquet(prediction_path))
    frontier = pareto_frontier(metrics)
    summary = {
        "scenario_exploration": True,
        "not_hyperparameter_tuning": True,
        "policy": "stockwise",
        "scenario_count": int(len(metrics)),
        "service_levels": list(SERVICE_LEVELS),
        "safety_stock_multipliers": list(SAFETY_STOCK_MULTIPLIERS),
        "note": "M5 inventory results are scenario simulations, not observed inventory outcomes.",
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metrics.to_csv(output / "scenario_metrics.csv", index=False)
    frontier.to_csv(output / "pareto_frontier.csv", index=False)
    return summary
