import pytest

from stockwise.forecasting import RollingOriginSplit, rolling_origin_splits


def test_rolling_origin_uses_expanding_training_window() -> None:
    splits = rolling_origin_splits(100, horizon=20, min_train_size=40, step=20)
    assert splits == [
        RollingOriginSplit(0, 40, 40, 60),
        RollingOriginSplit(0, 60, 60, 80),
        RollingOriginSplit(0, 80, 80, 100),
    ]


def test_rolling_origin_rejects_short_history() -> None:
    with pytest.raises(ValueError, match="not enough"):
        rolling_origin_splits(50, horizon=28, min_train_size=30)

