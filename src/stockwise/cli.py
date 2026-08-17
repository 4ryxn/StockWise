"""Small command-line entry point used for local smoke tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stockwise.analysis import write_eda_artifacts
from stockwise.data import build_store_dataset, validate_m5_raw_files
from stockwise.forecasting import seasonal_naive_forecast
from stockwise.inventory import InventoryScenario, recommend_order


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
