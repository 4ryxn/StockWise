"""Leakage-safe demand features for the future global forecasting model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURE_INPUT_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "date",
    "day_index",
    "units_sold",
    "weekday",
    "wday",
    "month",
    "year",
    "is_event",
    "snap_CA",
]
LAG_FEATURES = (1, 7, 14, 28)
ROLLING_FEATURES = (7, 28)
GENERATED_FEATURES = (
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
)


def load_feature_input(processed_directory: str | Path) -> pd.DataFrame:
    """Load only the target and known-in-advance fields needed for feature creation."""
    processed_path = Path(processed_directory)
    partitions = sorted(processed_path.glob("part-*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {processed_path}")
    return pd.concat(
        [pd.read_parquet(partition, columns=FEATURE_INPUT_COLUMNS) for partition in partitions],
        ignore_index=True,
    )


def build_leakage_safe_features(sales: pd.DataFrame) -> pd.DataFrame:
    """Create per-item features using demand known strictly before each item-day.

    Rolling standard deviation uses Pandas' sample standard deviation (``ddof=1``).
    ``sell_price`` is deliberately excluded because future prices are unavailable at inference.
    """
    missing_columns = sorted(set(FEATURE_INPUT_COLUMNS) - set(sales.columns))
    if missing_columns:
        raise ValueError(f"Sales data is missing required columns: {', '.join(missing_columns)}")
    if sales.duplicated(["item_id", "day_index"]).any():
        raise ValueError("Sales data contains duplicate item-day rows")

    features = (
        sales.sort_values(["item_id", "day_index"], kind="stable").reset_index(drop=True).copy()
    )
    grouped_target = features.groupby("item_id", sort=False)["units_sold"]
    for lag in LAG_FEATURES:
        features[f"lag_{lag}"] = grouped_target.shift(lag).astype("float32")

    prior_target = grouped_target.shift(1)
    prior_by_item = prior_target.groupby(features["item_id"], sort=False)
    for window in ROLLING_FEATURES:
        rolling_mean = prior_by_item.rolling(window, min_periods=window).mean()
        features[f"rolling_mean_{window}"] = (
            rolling_mean.reset_index(level=0, drop=True).astype("float32")
        )
    rolling_std = prior_by_item.rolling(28, min_periods=28).std()
    features["rolling_std_28"] = rolling_std.reset_index(level=0, drop=True).astype("float32")

    features["wday"] = features["wday"].astype("int8")
    features["month"] = features["month"].astype("int8")
    features["year"] = features["year"].astype("int16")
    features["snap_CA"] = features["snap_CA"].astype("int8")
    return features


def write_feature_profile(
    processed_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Write a compact feature-completeness profile without materializing a duplicate dataset."""
    features = build_leakage_safe_features(load_feature_input(processed_directory))
    profile = {
        "row_count": int(len(features)),
        "item_count": int(features["item_id"].nunique()),
        "minimum_date": pd.Timestamp(features["date"].min()).date().isoformat(),
        "maximum_date": pd.Timestamp(features["date"].max()).date().isoformat(),
        "zero_demand_rate": float((features["units_sold"] == 0).mean()),
        "price_feature_used": False,
        "generated_features": {
            column: {
                "dtype": str(features[column].dtype),
                "missing_count": int(features[column].isna().sum()),
                "missing_rate": float(features[column].isna().mean()),
            }
            for column in GENERATED_FEATURES
        },
    }
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    return profile
