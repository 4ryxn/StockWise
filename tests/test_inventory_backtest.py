import numpy as np

from stockwise.inventory_backtest import simulate_item_policy


def test_stockwise_first_order_does_not_depend_on_future_actual_demand() -> None:
    predictions = np.array([5.0] * 14)
    history = np.array([4.0] * 28)
    first = simulate_item_policy(np.array([1.0] * 14), predictions, history, "stockwise")
    changed = simulate_item_policy(
        np.array([1.0, 999.0] + [2.0] * 12), predictions, history, "stockwise"
    )
    assert first["orders"][0] == changed["orders"][0]


def test_lost_sales_and_lead_time_are_simulated() -> None:
    result = simulate_item_policy(
        np.array([10.0] * 8), np.zeros(8), np.zeros(28), "fixed_historical"
    )
    assert result["stockout_units"] > 0
    assert result["stockout_days"] > 0
