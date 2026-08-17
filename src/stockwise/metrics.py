"""Forecast evaluation metrics used throughout StockWise."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _paired_arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual_array.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("actual and predicted must contain only finite values")

    return actual_array, predicted_array


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return mean absolute error."""
    actual_array, predicted_array = _paired_arrays(actual, predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return root mean squared error."""
    actual_array, predicted_array = _paired_arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_array - predicted_array))))


def wape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return weighted absolute percentage error as a fraction.

    A value of 0.12 represents 12 percent aggregate absolute error.
    """
    actual_array, predicted_array = _paired_arrays(actual, predicted)
    denominator = float(np.sum(np.abs(actual_array)))
    if denominator == 0:
        raise ValueError("WAPE is undefined when total absolute actual demand is zero")
    return float(np.sum(np.abs(actual_array - predicted_array)) / denominator)

