import pandas as pd

from stockwise.inventory_sensitivity import pareto_frontier


def test_pareto_frontier_excludes_dominated_service_inventory_choice() -> None:
    metrics = pd.DataFrame(
        {
            "service_level": [0.9, 0.95, 0.99],
            "safety_stock_multiplier": [1, 1, 1],
            "fill_rate": [0.90, 0.95, 0.94],
            "average_on_hand_units": [10.0, 12.0, 14.0],
        }
    )
    frontier = pareto_frontier(metrics)
    assert frontier["service_level"].tolist() == [0.9, 0.95]
