import json

import pandas as pd
import pytest

from stockwise.data import build_store_dataset, transform_sales_chunk, validate_m5_raw_files


@pytest.fixture
def tiny_m5_directory(tmp_path):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()

    pd.DataFrame(
        {
            "id": ["ITEM_1_CA_1_evaluation", "ITEM_2_TX_1_evaluation"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
            "store_id": ["CA_1", "TX_1"],
            "state_id": ["CA", "TX"],
            "d_1": [2, 4],
            "d_2": [3, 5],
            "d_3": [1, 6],
        }
    ).to_csv(raw_directory / "sales_train_evaluation.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11101],
            "weekday": ["Saturday", "Sunday", "Monday"],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "d": ["d_1", "d_2", "d_3"],
            "event_name_1": [None, "Event", None],
            "event_type_1": [None, "Cultural", None],
            "event_name_2": [None, None, None],
            "event_type_2": [None, None, None],
            "snap_CA": [0, 0, 1],
            "snap_TX": [0, 0, 0],
            "snap_WI": [0, 0, 0],
        }
    ).to_csv(raw_directory / "calendar.csv", index=False)

    pd.DataFrame(
        {
            "store_id": ["CA_1", "TX_1"],
            "item_id": ["ITEM_1", "ITEM_2"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [2.5, 3.5],
        }
    ).to_csv(raw_directory / "sell_prices.csv", index=False)
    return raw_directory


def test_validate_m5_raw_files_reports_store_and_day_coverage(tiny_m5_directory) -> None:
    report = validate_m5_raw_files(tiny_m5_directory, store_id="CA_1")
    assert report.day_column_count == 3
    assert report.first_day == "d_1"
    assert report.last_day == "d_3"
    assert report.store_series_count == 1


def test_validate_m5_raw_files_lists_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="calendar.csv"):
        validate_m5_raw_files(tmp_path)


def test_transform_sales_chunk_creates_item_day_rows(tiny_m5_directory) -> None:
    sales = pd.read_csv(tiny_m5_directory / "sales_train_evaluation.csv").query(
        "store_id == 'CA_1'"
    )
    calendar = pd.read_csv(tiny_m5_directory / "calendar.csv")
    prices = pd.read_csv(tiny_m5_directory / "sell_prices.csv")

    transformed = transform_sales_chunk(sales, calendar, prices)

    assert len(transformed) == 3
    assert transformed["units_sold"].tolist() == [2, 3, 1]
    assert transformed["day_index"].tolist() == [1, 2, 3]
    assert transformed["sell_price"].tolist() == [2.5, 2.5, 2.5]
    assert transformed["is_event"].tolist() == [False, True, False]


def test_build_store_dataset_writes_parquet_and_quality_report(tiny_m5_directory, tmp_path) -> None:
    pytest.importorskip("pyarrow")
    output_directory = tmp_path / "processed"
    report = build_store_dataset(
        tiny_m5_directory,
        output_directory,
        store_id="CA_1",
        item_chunk_size=1,
    )

    assert report.partition_count == 1
    assert report.row_count == 3
    assert report.item_count == 1
    assert report.missing_price_rate == 0
    assert (output_directory / "part-0000.parquet").is_file()
    saved_report = json.loads((output_directory / "data_quality_report.json").read_text())
    assert saved_report["row_count"] == 3

