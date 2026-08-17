import json

import pandas as pd
import pytest

from stockwise.analysis import create_eda_summary, write_eda_artifacts


@pytest.fixture
def synthetic_sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["item_a", "item_a", "item_b", "item_b", "item_c"],
            "dept_id": ["dept_1", "dept_1", "dept_1", "dept_1", "dept_2"],
            "cat_id": ["cat_1", "cat_1", "cat_1", "cat_1", "cat_2"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-01", "2021-01-01"]
            ),
            "year": [2020, 2020, 2020, 2020, 2021],
            "units_sold": [0, 5, 3, 3, 2],
            "sell_price": [None, 1.0, None, None, 2.0],
        }
    )


def test_create_eda_summary_calculates_headline_metrics(synthetic_sales) -> None:
    summary, breakdowns = create_eda_summary(synthetic_sales)

    assert summary["row_count"] == 5
    assert summary["item_count"] == 3
    assert summary["minimum_date"] == "2020-01-01"
    assert summary["maximum_date"] == "2021-01-01"
    assert summary["total_units_sold"] == 13
    assert summary["zero_demand_rate"] == pytest.approx(0.2)
    assert summary["duplicate_item_date_count"] == 1
    assert summary["missing_price_rate"] == pytest.approx(0.6)
    assert summary["missing_price_rate_when_units_sold_positive"] == pytest.approx(0.5)
    assert summary["top_10_items_by_total_units_sold"][0] == {
        "item_id": "item_b",
        "total_units_sold": 6,
    }
    assert summary["missing_price_by_category"] == [
        {"cat_id": "cat_1", "row_count": 4, "missing_price_rows": 3, "missing_price_rate": 0.75},
        {"cat_id": "cat_2", "row_count": 1, "missing_price_rows": 0, "missing_price_rate": 0.0},
    ]
    assert breakdowns["missing_price_by_year"].to_dict(orient="records") == [
        {"year": 2020, "row_count": 4, "missing_price_rows": 3, "missing_price_rate": 0.75},
        {"year": 2021, "row_count": 1, "missing_price_rows": 0, "missing_price_rate": 0.0},
    ]


def test_write_eda_artifacts_reads_partitions_and_writes_reports(tmp_path, synthetic_sales) -> None:
    pytest.importorskip("pyarrow")
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    synthetic_sales.iloc[:3].to_parquet(processed_directory / "part-0000.parquet", index=False)
    synthetic_sales.iloc[3:].to_parquet(processed_directory / "part-0001.parquet", index=False)
    output_directory = tmp_path / "eda"

    summary = write_eda_artifacts(processed_directory, output_directory)

    assert summary["row_count"] == 5
    assert json.loads((output_directory / "summary.json").read_text())["item_count"] == 3
    assert (output_directory / "missing_price_by_year.csv").is_file()
    assert (output_directory / "missing_price_by_category.csv").is_file()
    assert (output_directory / "missing_price_by_department.csv").is_file()
    assert (output_directory / "top_10_items_by_total_units_sold.csv").is_file()
