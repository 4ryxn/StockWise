"""Simple forecasting baselines that every learned model must beat."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def seasonal_naive_forecast(
    history: ArrayLike,
    horizon: int,
    season_length: int = 7,
) -> NDArray[np.float64]:
    """Repeat the most recent complete season for the requested horizon."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 1:
        raise ValueError("history must be one-dimensional")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    if len(values) < season_length:
        raise ValueError("history must contain at least one complete season")
    if not np.isfinite(values).all():
        raise ValueError("history must contain only finite values")

    last_season = values[-season_length:]
    return np.resize(last_season, horizon).astype(float)

