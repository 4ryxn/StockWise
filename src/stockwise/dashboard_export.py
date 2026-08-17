"""Compact static-data export for the frontend dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def rows(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_dashboard_data(artifacts_directory: str | Path, output_directory: str | Path) -> dict:
    a = Path(artifacts_directory)
    def load(path: Path) -> dict:
        return json.loads(path.read_text())
    e = load(a / "eda/summary.json")
    lightgbm = load(a / "lightgbm/summary.json")
    inventory = load(a / "inventory/summary.json")
    sensitivity = load(a / "inventory_sensitivity/summary.json")
    data = {
        "overview": {
            "row_count": e["row_count"],
            "item_count": e["item_count"],
        "wape": lightgbm["all_folds_combined"]["wape"],
        "improvement": lightgbm["baseline_comparison"]["wape_relative_improvement_percent"],
        },
        "folds": lightgbm["folds"],
        "categories": rows(a / "lightgbm/metrics_by_category.csv"),
        "departments": rows(a / "lightgbm/metrics_by_department.csv"),
        "importance": rows(a / "lightgbm/feature_importance.csv"),
        "inventory": inventory["policies"],
        "sensitivity": rows(a / "inventory_sensitivity/scenario_metrics.csv"),
        "pareto": rows(a / "inventory_sensitivity/pareto_frontier.csv"),
        "methodology": [
            "28-day rolling held-out backtests across M5 CA_1.",
            "LightGBM uses lag, rolling, ID, and known calendar features only.",
            "Forecasts are backtest results, not live predictions.",
            "Inventory is a lost-sales scenario simulation; M5 has no observed inventory.",
        ],
        "note": sensitivity["note"],
    }
    o = Path(output_directory)
    o.mkdir(parents=True, exist_ok=True)
    (o / "dashboard.json").write_text(json.dumps(data, indent=2) + "\n")
    return data
