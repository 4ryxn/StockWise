"""Leakage-safe rolling-origin validation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingOriginSplit:
    """Half-open positional slices for a time-series validation fold."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int


def rolling_origin_splits(
    n_observations: int,
    horizon: int,
    min_train_size: int,
    step: int | None = None,
) -> list[RollingOriginSplit]:
    """Create expanding-window validation folds using positional boundaries."""
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be positive")
    if min_train_size + horizon > n_observations:
        raise ValueError("not enough observations for one validation fold")

    resolved_step = horizon if step is None else step
    if resolved_step <= 0:
        raise ValueError("step must be positive")

    splits: list[RollingOriginSplit] = []
    train_end = min_train_size
    while train_end + horizon <= n_observations:
        splits.append(
            RollingOriginSplit(
                train_start=0,
                train_end=train_end,
                validation_start=train_end,
                validation_end=train_end + horizon,
            )
        )
        train_end += resolved_step
    return splits

