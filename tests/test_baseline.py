import pytest

from stockwise.forecasting import seasonal_naive_forecast


def test_seasonal_naive_repeats_last_complete_season() -> None:
    result = seasonal_naive_forecast([1, 2, 3, 4, 5, 6, 7], horizon=10, season_length=7)
    assert result.tolist() == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]


def test_seasonal_naive_requires_one_season() -> None:
    with pytest.raises(ValueError, match="complete season"):
        seasonal_naive_forecast([1, 2], horizon=3, season_length=7)

