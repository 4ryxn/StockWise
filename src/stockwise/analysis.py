"""Reproducible exploratory data-quality summaries for processed sales data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ANALYSIS_COLUMNS = ["item_id", "dept_id", "cat_id", "date", "year", "units_sold", "sell_price"]


def load_processed_sales(processed_directory: str | Path) -> pd.DataFrame:
    """Load the analysis columns from every Parquet partition in a processed dataset."""
    processed_path = Path(processed_directory)
    partitions = sorted(processed_path.glob("part-*.parquet"))
    if not partitions:
        raise FileNotFoundError(f"No Parquet partitions found in {processed_path}")
    return pd.concat(
        [pd.read_parquet(partition, columns=ANALYSIS_COLUMNS) for partition in partitions],
        ignore_index=True,
    )


def _missing_price_breakdown(sales: pd.DataFrame, group_column: str) -> pd.DataFrame:
    missing_price = sales["sell_price"].isna()
    breakdown = (
        sales.assign(missing_price=missing_price)
        .groupby(group_column, as_index=False, dropna=False)
        .agg(
            row_count=("item_id", "size"),
            missing_price_rows=("missing_price", "sum"),
        )
    )
    breakdown["missing_price_rows"] = breakdown["missing_price_rows"].astype("int64")
    breakdown["missing_price_rate"] = breakdown["missing_price_rows"] / breakdown["row_count"]
    return breakdown.sort_values(group_column, kind="stable").reset_index(drop=True)


def create_eda_summary(sales: pd.DataFrame) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Calculate headline EDA measures and CSV-ready data-quality breakdowns."""
    missing_columns = sorted(set(ANALYSIS_COLUMNS) - set(sales.columns))
    if missing_columns:
        raise ValueError(f"Sales data is missing required columns: {', '.join(missing_columns)}")
    if sales.empty:
        raise ValueError("Sales data must contain at least one row")

    missing_price = sales["sell_price"].isna()
    positive_demand = sales["units_sold"] > 0
    positive_demand_count = int(positive_demand.sum())
    top_items = (
        sales.groupby("item_id", as_index=False)["units_sold"]
        .sum()
        .rename(columns={"units_sold": "total_units_sold"})
        .sort_values(["total_units_sold", "item_id"], ascending=[False, True], kind="stable")
        .head(10)
        .reset_index(drop=True)
    )

    breakdowns = {
        "missing_price_by_year": _missing_price_breakdown(sales, "year"),
        "missing_price_by_category": _missing_price_breakdown(sales, "cat_id"),
        "missing_price_by_department": _missing_price_breakdown(sales, "dept_id"),
        "top_10_items_by_total_units_sold": top_items,
    }
    summary: dict[str, object] = {
        "row_count": int(len(sales)),
        "item_count": int(sales["item_id"].nunique()),
        "minimum_date": pd.Timestamp(sales["date"].min()).date().isoformat(),
        "maximum_date": pd.Timestamp(sales["date"].max()).date().isoformat(),
        "total_units_sold": int(sales["units_sold"].sum()),
        "zero_demand_rate": float((sales["units_sold"] == 0).mean()),
        "duplicate_item_date_count": int(sales.duplicated(["item_id", "date"]).sum()),
        "missing_price_rate": float(missing_price.mean()),
        "missing_price_rate_when_units_sold_positive": (
            float(missing_price[positive_demand].mean()) if positive_demand_count else None
        ),
        "missing_price_by_year": breakdowns["missing_price_by_year"].to_dict(orient="records"),
        "missing_price_by_category": breakdowns["missing_price_by_category"].to_dict(
            orient="records"
        ),
        "missing_price_by_department": breakdowns["missing_price_by_department"].to_dict(
            orient="records"
        ),
        "top_10_items_by_total_units_sold": top_items.to_dict(orient="records"),
    }
    return summary, breakdowns


def write_eda_artifacts(
    processed_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Create a JSON summary and CSV breakdowns without modifying processed data."""
    sales = load_processed_sales(processed_directory)
    summary, breakdowns = create_eda_summary(sales)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, breakdown in breakdowns.items():
        breakdown.to_csv(output_path / f"{name}.csv", index=False)
    return summary
