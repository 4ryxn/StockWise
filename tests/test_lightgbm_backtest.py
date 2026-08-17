import numpy as np
import pandas as pd

from stockwise.forecasting.lightgbm_backtest import (
    _category_maps,
    recursive_forecast,
)


class LagOneModel:
    def predict(self, features):
        return features["lag_1"].to_numpy()


def test_recursive_forecast_does_not_use_future_actuals() -> None:
    history = pd.DataFrame(
        {"item_id": ["a"] * 28, "day_index": range(1, 29), "units_sold": range(1, 29)}
    )
    future = pd.DataFrame(
        {
            "item_id": ["a", "a"],
            "day_index": [29, 30],
            "units_sold": [999, 777],
            "cat_id": ["c", "c"],
            "dept_id": ["d", "d"],
            "weekday": ["Mon", "Tue"],
            "wday": [1, 2],
            "month": [1, 1],
            "year": [2020, 2020],
            "is_event": [False, False],
            "snap_CA": [0, 0],
        }
    )
    source = pd.concat([history.assign(cat_id="c", dept_id="d", weekday="Sun"), future])
    maps = _category_maps(source)
    first = recursive_forecast(LagOneModel(), history, future, maps)
    future.loc[:, "units_sold"] = [1, 2]
    second = recursive_forecast(LagOneModel(), history, future, maps)
    assert np.array_equal(first, second)
    assert first.tolist() == [[28.0, 28.0]]
