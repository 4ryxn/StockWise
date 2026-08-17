"""Small command-line entry point used for local smoke tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stockwise.analysis import write_eda_artifacts
from stockwise.data import build_store_dataset, validate_m5_raw_files
from stockwise.forecasting import (
    seasonal_naive_forecast,
    write_baseline_artifacts,
    write_feature_profile,
    write_lightgbm_artifacts,
)
from stockwise.inventory import InventoryScenario, recommend_order
from stockwise.inventory_backtest import write_inventory_artifacts
from stockwise.inventory_sensitivity import write_inventory_sensitivity_artifacts


def run_demo() -> None:
    history = [8, 9, 7, 10, 12, 15, 13, 9, 10, 8, 11, 13, 16, 14]
    forecast = seasonal_naive_forecast(history, horizon=7).tolist()
    recommendation = recommend_order(
        InventoryScenario(
            forecast_daily_mean=sum(forecast) / len(forecast),
            forecast_daily_std=2.5,
            lead_time_days=3,
            review_period_days=7,
            service_level=0.95,
            on_hand_units=20,
        )
    )
    print({"seven_day_forecast": forecast, "inventory": asdict(recommendation)})


def main() -> None:
    parser = argparse.ArgumentParser(description="StockWise command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="Run a dependency-free smoke-test demonstration")

    validate_parser = subparsers.add_parser("validate-m5", help="Validate raw M5 CSV files")
    validate_parser.add_argument("raw_dir", type=Path)
    validate_parser.add_argument("--store-id", default="CA_1")

    build_parser = subparsers.add_parser("build-m5", help="Build a store-level Parquet dataset")
    build_parser.add_argument("raw_dir", type=Path)
    build_parser.add_argument("output_dir", type=Path)
    build_parser.add_argument("--store-id", default="CA_1")
    build_parser.add_argument("--chunk-size", type=int, default=250)

    eda_parser = subparsers.add_parser("eda", help="Create EDA and data-quality artifacts")
    eda_parser.add_argument("processed_dir", type=Path)
    eda_parser.add_argument("output_dir", type=Path)

    baseline_parser = subparsers.add_parser(
        "backtest-baseline", help="Evaluate the seasonal-naive baseline on fixed validation folds"
    )
    baseline_parser.add_argument("processed_dir", type=Path)
    baseline_parser.add_argument("output_dir", type=Path)

    feature_profile_parser = subparsers.add_parser(
        "feature-profile", help="Create a lightweight leakage-safe feature profile"
    )
    feature_profile_parser.add_argument("processed_dir", type=Path)
    feature_profile_parser.add_argument("output_dir", type=Path)

    lightgbm_parser = subparsers.add_parser(
        "backtest-lightgbm", help="Run the global LightGBM backtest"
    )
    lightgbm_parser.add_argument("processed_dir", type=Path)
    lightgbm_parser.add_argument("output_dir", type=Path)

    inventory_parser = subparsers.add_parser(
        "backtest-inventory", help="Run inventory scenario simulation"
    )
    inventory_parser.add_argument("processed_dir", type=Path)
    inventory_parser.add_argument("prediction_path", type=Path)
    inventory_parser.add_argument("output_dir", type=Path)

    sensitivity_parser = subparsers.add_parser(
        "inventory-sensitivity", help="Explore inventory scenarios"
    )
    sensitivity_parser.add_argument("processed_dir", type=Path)
    sensitivity_parser.add_argument("prediction_path", type=Path)
    sensitivity_parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()
    if args.command == "demo":
        run_demo()
    elif args.command == "validate-m5":
        print(json.dumps(asdict(validate_m5_raw_files(args.raw_dir, args.store_id)), indent=2))
    elif args.command == "build-m5":
        report = build_store_dataset(
            args.raw_dir,
            args.output_dir,
            store_id=args.store_id,
            item_chunk_size=args.chunk_size,
        )
        print(json.dumps(asdict(report), indent=2))
    elif args.command == "eda":
        print(json.dumps(write_eda_artifacts(args.processed_dir, args.output_dir), indent=2))
    elif args.command == "backtest-baseline":
        print(json.dumps(write_baseline_artifacts(args.processed_dir, args.output_dir), indent=2))
    elif args.command == "feature-profile":
        print(json.dumps(write_feature_profile(args.processed_dir, args.output_dir), indent=2))
    elif args.command == "backtest-lightgbm":
        print(json.dumps(write_lightgbm_artifacts(args.processed_dir, args.output_dir), indent=2))
    elif args.command == "backtest-inventory":
        print(
            json.dumps(
                write_inventory_artifacts(
                    args.processed_dir, args.prediction_path, args.output_dir
                ),
                indent=2,
            )
        )
    elif args.command == "inventory-sensitivity":
        print(
            json.dumps(
                write_inventory_sensitivity_artifacts(
                    args.processed_dir, args.prediction_path, args.output_dir
                ),
                indent=2,
            )
        )
