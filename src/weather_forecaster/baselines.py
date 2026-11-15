"""Forecasts that need no model. A neural network has to beat these to be worth anything."""

from __future__ import annotations

import numpy as np


def persistence(truth: np.ndarray, lead_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Forecast that the weather stays exactly as it is now.

    Returns (forecast, target): the field at each start time, paired with the field
    lead_steps later. Start times too close to the end to have a target are dropped.
    """
    if lead_steps < 1:
        raise ValueError("lead_steps must be at least 1")
    return truth[:-lead_steps], truth[lead_steps:]


def climatology(train: np.ndarray) -> np.ndarray:
    """The long-term average field: one (latitude, longitude) map."""
    return train.mean(axis=0, dtype=np.float64)


def iso_week(times: np.ndarray) -> np.ndarray:
    """ISO calendar week number (1 to 53) of each datetime64."""
    days, inverse = np.unique(times.astype("datetime64[D]"), return_inverse=True)
    weeks = np.array([day.astype(object).isocalendar()[1] for day in days])
    return weeks[inverse]


def weekly_climatology(train: np.ndarray, train_times: np.ndarray) -> np.ndarray:
    """Average field for each calendar week, shape (54, latitude, longitude).

    Indexed by ISO week number, so index 0 is unused.
    """
    weeks = iso_week(train_times)
    fields = np.zeros((54, *train.shape[1:]))
    for week in range(1, 54):
        fields[week] = train[weeks == week].mean(axis=0, dtype=np.float64)
    return fields
