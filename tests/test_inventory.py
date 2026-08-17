import math

import pytest

from stockwise.inventory import InventoryScenario, recommend_order


def test_low_inventory_produces_reorder_recommendation() -> None:
    result = recommend_order(
        InventoryScenario(
            forecast_daily_mean=10,
            forecast_daily_std=2,
            lead_time_days=3,
            review_period_days=7,
            service_level=0.95,
            on_hand_units=20,
        )
    )

    assert result.should_reorder is True
    assert result.inventory_position == 20
    assert result.safety_stock == pytest.approx(1.6448536269514722 * 2 * math.sqrt(3))
    assert result.recommended_order_units > 0


def test_high_inventory_does_not_trigger_order() -> None:
    result = recommend_order(
        InventoryScenario(
            forecast_daily_mean=10,
            forecast_daily_std=2,
            lead_time_days=3,
            review_period_days=7,
            service_level=0.95,
            on_hand_units=200,
        )
    )
    assert result.should_reorder is False
    assert result.recommended_order_units == 0


def test_inventory_scenario_rejects_invalid_service_level() -> None:
    with pytest.raises(ValueError, match="service_level"):
        InventoryScenario(
            forecast_daily_mean=10,
            forecast_daily_std=2,
            lead_time_days=3,
            review_period_days=7,
            service_level=1.0,
            on_hand_units=20,
        )

