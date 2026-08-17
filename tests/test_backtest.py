import pandas as pd
import pytest

from stockwise.forecasting.backtest import (
    BASELINE_FOLDS,
    BaselineFold,
    backtest_seasonal_naive,
    write_baseline_artifacts,
)


def _synthetic_demand() -> pd.DataFrame:
    rows = []
    for item_id, category, department, offset in [
        ("item_a", "cat_a", "dept_a", 0),
        ("item_b", "cat_b", "dept_b", 10),
    ]:
        for day_index in range(1, 43):
            rows.append(
                {
                    "item_id": item_id,
                    "cat_id": category,
                    "dept_id": department,
                    "day_index": day_index,
                    "units_sold": (day_index + offset) % 7,
                }
            )
    return pd.DataFrame(rows)


def test_default_baseline_folds_match_the_frozen_validation_windows() -> None:
    assert BASELINE_FOLDS == (
        BaselineFold("fold_1", 1, 1857, 1858, 1885),
        BaselineFold("fold_2", 1, 1885, 1886, 1913),
        BaselineFold("fold_3", 1, 1913, 1914, 1941),
    )


def test_backtest_seasonal_naive_reports_overall_and_segment_metrics() -> None:
    folds = (BaselineFold("fold_1", 1, 28, 29, 35), BaselineFold("fold_2", 1, 35, 36, 42))
    summary, fold_metrics, category_metrics, department_metrics = backtest_seasonal_naive(
        _synthetic_demand(), folds=folds, horizon=7
    )

    assert [fold["name"] for fold in summary["folds"]] == ["fold_1", "fold_2"]
    assert summary["all_folds_combined"]["wape"] == 0
    assert summary["all_folds_combined"]["mae"] == 0
    assert summary["all_folds_combined"]["rmse"] == 0
    assert fold_metrics["row_count"].tolist() == [14, 14, 28]
    assert fold_metrics["name"].tolist() == ["fold_1", "fold_2", "all_folds"]
    assert set(category_metrics["fold"]) == {"fold_1", "fold_2", "all_folds"}
    assert set(department_metrics["dept_id"]) == {"dept_a", "dept_b"}


def test_backtest_rejects_duplicate_item_day_rows() -> None:
    demand = _synthetic_demand()
    duplicated = pd.concat([demand, demand.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        backtest_seasonal_naive(duplicated, folds=(BaselineFold("fold", 1, 28, 29, 35),), horizon=7)


def test_write_baseline_artifacts_creates_summary_and_csv_files(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    demand = _synthetic_demand()
    demand.to_parquet(processed_directory / "part-0000.parquet", index=False)

    output_directory = tmp_path / "baseline"
    folds = (
        BaselineFold("fold_1", 1, 28, 29, 35),
        BaselineFold("fold_2", 1, 35, 36, 42),
    )
    summary = write_baseline_artifacts(
        processed_directory, output_directory, folds=folds, horizon=7
    )

    assert summary["all_folds_combined"]["mae"] == 0
    assert (output_directory / "summary.json").is_file()
    assert (output_directory / "fold_metrics.csv").is_file()
    assert (output_directory / "metrics_by_category.csv").is_file()
    assert (output_directory / "metrics_by_department.csv").is_file()
