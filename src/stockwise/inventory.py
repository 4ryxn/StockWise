"""Transparent scenario-based inventory policy calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class InventoryScenario:
    forecast_daily_mean: float
    forecast_daily_std: float
    lead_time_days: int
    review_period_days: int
    service_level: float
    on_hand_units: float
    on_order_units: float = 0.0
    backorders_units: float = 0.0

    def __post_init__(self) -> None:
        nonnegative_values = {
            "forecast_daily_mean": self.forecast_daily_mean,
            "forecast_daily_std": self.forecast_daily_std,
            "on_hand_units": self.on_hand_units,
            "on_order_units": self.on_order_units,
            "backorders_units": self.backorders_units,
        }
        for name, value in nonnegative_values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.lead_time_days <= 0:
            raise ValueError("lead_time_days must be positive")
        if self.review_period_days <= 0:
            raise ValueError("review_period_days must be positive")
        if not 0.5 < self.service_level < 1:
            raise ValueError("service_level must be between 0.5 and 1")


@dataclass(frozen=True)
class InventoryRecommendation:
    inventory_position: float
    safety_stock: float
    reorder_point: float
    order_up_to_level: float
    recommended_order_units: int
    should_reorder: bool


def recommend_order(scenario: InventoryScenario) -> InventoryRecommendation:
    """Apply a continuous-review reorder-point and order-up-to policy.

    Daily forecast errors are assumed independent and normally distributed. This is a
    documented MVP assumption that will later be checked through backtesting.
    """
    z_score = NormalDist().inv_cdf(scenario.service_level)
    lead_time_std = scenario.forecast_daily_std * math.sqrt(scenario.lead_time_days)
    protection_days = scenario.lead_time_days + scenario.review_period_days
    protection_std = scenario.forecast_daily_std * math.sqrt(protection_days)

    safety_stock = z_score * lead_time_std
    reorder_point = scenario.forecast_daily_mean * scenario.lead_time_days + safety_stock
    order_up_to_level = (
        scenario.forecast_daily_mean * protection_days + z_score * protection_std
    )
    inventory_position = (
        scenario.on_hand_units + scenario.on_order_units - scenario.backorders_units
    )
    should_reorder = inventory_position <= reorder_point
    recommended_units = (
        math.ceil(max(0.0, order_up_to_level - inventory_position)) if should_reorder else 0
    )

    return InventoryRecommendation(
        inventory_position=inventory_position,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        order_up_to_level=order_up_to_level,
        recommended_order_units=recommended_units,
        should_reorder=should_reorder,
    )

