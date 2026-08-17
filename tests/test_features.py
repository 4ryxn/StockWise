import math

import pandas as pd
import pytest

from stockwise.forecasting.features import (
    GENERATED_FEATURES,
    build_leakage_safe_features,
    write_feature_profile,
)


def _sales_row(item_id: str, day_index: int, units_sold: int) -> dict[str, object]:
    return {
        "item_id": item_id,
        "dept_id": f"dept_{item_id}",
        "cat_id": f"cat_{item_id}",
        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(day_index - 1, unit="D"),
        "day_index": day_index,
        "units_sold": units_sold,
        "weekday": "Wednesday",
        "wday": 1,
        "month": 1,
        "year": 2020,
        "is_event": False,
        "snap_CA": 0,
    }


def _synthetic_sales() -> pd.DataFrame:
    rows = [_sales_row("item_a", day, day) for day in range(1, 31)]
    rows.extend(_sales_row("item_b", day, 0 if day < 3 else 5) for day in range(1, 31))
    return pd.DataFrame(rows)


def test_features_have_exact_lag_and_rolling_values() -> None:
    features = build_leakage_safe_features(_synthetic_sales())
    day_29 = features.loc[(features["item_id"] == "item_a") & (features["day_index"] == 29)].iloc[0]

    assert day_29["lag_1"] == 28
    assert day_29["lag_7"] == 22
    assert day_29["lag_14"] == 15
    assert day_29["lag_28"] == 1
    assert day_29["rolling_mean_7"] == pytest.approx(25)
    assert day_29["rolling_mean_28"] == pytest.approx(14.5)
    assert day_29["rolling_std_28"] == pytest.approx(math.sqrt(67.66666666666667))
    assert all(str(features[column].dtype) == "float32" for column in GENERATED_FEATURES)


def test_current_target_cannot_change_its_own_features() -> None:
    original = _synthetic_sales()
    changed = original.copy()
    changed.loc[(changed["item_id"] == "item_a") & (changed["day_index"] == 29), "units_sold"] = 999

    original_features = build_leakage_safe_features(original)
    changed_features = build_leakage_safe_features(changed)
    original_day = original_features.loc[
        (original_features["item_id"] == "item_a") & (original_features["day_index"] == 29)
    ].iloc[0]
    changed_day = changed_features.loc[
        (changed_features["item_id"] == "item_a") & (changed_features["day_index"] == 29)
    ].iloc[0]

    assert original_day[list(GENERATED_FEATURES)].equals(changed_day[list(GENERATED_FEATURES)])


def test_item_boundaries_and_zero_demand_history_are_preserved() -> None:
    features = build_leakage_safe_features(_synthetic_sales())
    first_item_b = features.loc[
        (features["item_id"] == "item_b") & (features["day_index"] == 1)
    ].iloc[0]
    day_8_item_b = features.loc[
        (features["item_id"] == "item_b") & (features["day_index"] == 8)
    ].iloc[0]

    assert len(features) == 60
    assert pd.isna(first_item_b["lag_1"])
    assert day_8_item_b["lag_1"] == 5
    assert day_8_item_b["rolling_mean_7"] == pytest.approx(25 / 7)
    assert "sell_price" not in features.columns


def test_write_feature_profile_creates_only_json(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    processed_directory = tmp_path / "processed"
    processed_directory.mkdir()
    _synthetic_sales().to_parquet(processed_directory / "part-0000.parquet", index=False)

    output_directory = tmp_path / "features"
    profile = write_feature_profile(processed_directory, output_directory)

    assert profile["row_count"] == 60
    assert profile["price_feature_used"] is False
    assert (output_directory / "profile.json").is_file()
    assert list(output_directory.iterdir()) == [output_directory / "profile.json"]
