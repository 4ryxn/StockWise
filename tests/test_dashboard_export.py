import pandas as pd

from stockwise.dashboard_export import _select_planner_items


def test_planner_export_is_category_balanced_and_uses_only_pre_fold_history() -> None:
    prediction_rows = []
    history_rows = []
    for category in ("FOODS", "HOBBIES", "HOUSEHOLD"):
        for item_number in range(2):
            item_id = f"{category}_{item_number}"
            for day_index in (10, 11):
                prediction_rows.append(
                    {
                        "item_id": item_id,
                        "cat_id": category,
                        "dept_id": f"{category}_1",
                        "day_index": day_index,
                        "prediction": float(item_number + 1),
                    }
                )
            history_rows.extend(
                [
                    {"item_id": item_id, "day_index": 8, "units_sold": item_number},
                    {"item_id": item_id, "day_index": 9, "units_sold": item_number + 2},
                    {"item_id": item_id, "day_index": 10, "units_sold": 999},
                ]
            )

    data = _select_planner_items(
        pd.DataFrame(prediction_rows), pd.DataFrame(history_rows), items_per_category=1
    )

    assert data["forecast_fold"] == "fold_3"
    assert data["forecast_start_day"] == 10
    assert len(data["items"]) == 3
    assert {item["category"] for item in data["items"]} == {"FOODS", "HOBBIES", "HOUSEHOLD"}
    assert all(item["item_id"].endswith("_1") for item in data["items"])
    assert all(item["forecast"] == [2.0, 2.0] for item in data["items"])
    assert all(item["historical_daily_demand_std"] == 1.0 for item in data["items"])
