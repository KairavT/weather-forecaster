"""Turn downloaded ERA5 channels into normalized arrays a network can train on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weather_forecaster.era5 import LocalERA5


@dataclass(frozen=True)
class Normalizer:
    """Per-channel mean and standard deviation, measured on the training period only.

    Geopotential is around 54000 and temperature around 270, so without rescaling one
    channel would dominate the loss. Fields have shape (..., channel, latitude, longitude).
    """

    mean: np.ndarray  # (channel,)
    std: np.ndarray  # (channel,)

    def normalize(self, fields: np.ndarray) -> np.ndarray:
        return (fields - self.mean[:, None, None]) / self.std[:, None, None]

    def denormalize(self, fields: np.ndarray) -> np.ndarray:
        return fields * self.std[:, None, None] + self.mean[:, None, None]


def stack_channels(data: LocalERA5, channels: list[str], start: str, end: str) -> np.ndarray:
    """Load channels for a period as one float32 array (time, channel, latitude, longitude)."""
    period = data.period(start, end)
    return np.stack([data.channel(name)[period] for name in channels], axis=1)


def fit_normalizer(fields: np.ndarray) -> Normalizer:
    """Measure each channel's mean and spread over all time steps and grid points."""
    mean = fields.mean(axis=(0, 2, 3), dtype=np.float64)
    std = fields.std(axis=(0, 2, 3), dtype=np.float64)
    return Normalizer(mean.astype(np.float32), std.astype(np.float32))
