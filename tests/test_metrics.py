import math

import pytest

from stockwise.metrics import mae, rmse, wape


def test_metrics_return_expected_values() -> None:
    actual = [10, 20, 30]
    predicted = [12, 18, 33]

    assert mae(actual, predicted) == pytest.approx(7 / 3)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(17 / 3))
    assert wape(actual, predicted) == pytest.approx(7 / 60)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mae([1, 2], [1])


def test_wape_rejects_zero_total_actual() -> None:
    with pytest.raises(ValueError, match="undefined"):
        wape([0, 0], [1, 0])

