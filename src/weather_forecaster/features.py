"""Inputs the stepping model gets besides the weather itself: the clock, the calendar, the map.

The same atmosphere evolves differently at noon and at midnight, in January and in July, over
land and over sea. None of that can be read off the weather fields, so it is handed to the
network as extra input channels. All of it is known in advance for any future time, which is
what lets a forecast be stepped forward.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from weather_forecaster.era5 import LocalERA5

CLOCK_CHANNELS = 4  # sine and cosine of the local solar time, sine and cosine of the season
LATITUDE_CHANNELS = 2  # sine and cosine of latitude, which static_maps adds to the named fields


def forcing_channels(n_static: int) -> int:
    """How many channels Forcings produces when static_maps is given n_static field names."""
    return CLOCK_CHANNELS + n_static + LATITUDE_CHANNELS


def day_fraction(times: np.ndarray) -> np.ndarray:
    """How far through the UTC day each time is, from 0 at midnight up to 1."""
    times = times.astype("datetime64[h]")
    return (times - times.astype("datetime64[D]")).astype(np.float64) / 24


def year_fraction(times: np.ndarray) -> np.ndarray:
    """How far through its calendar year each time is, from 0 on 1 January up to 1."""
    times = times.astype("datetime64[h]")
    years = times.astype("datetime64[Y]")
    start = years.astype("datetime64[h]")
    next_year = (years + np.timedelta64(1, "Y")).astype("datetime64[h]")
    length = (next_year - start).astype(np.float64)
    return (times - start).astype(np.float64) / length


def static_maps(data: LocalERA5, names: list[str]) -> np.ndarray:
    """(n, latitude, longitude): each named field standardized, then sine and cosine of latitude.

    The sine tells the hemispheres apart, which the network needs to know what a date means:
    January is winter in one and summer in the other.
    """
    maps = []
    for name in names:
        field = data.static(name).astype(np.float64)
        maps.append((field - field.mean()) / field.std())
    latitude = np.deg2rad(data.latitude)[:, None] * np.ones(len(data.longitude))
    maps += [np.sin(latitude), np.cos(latitude)]
    return np.stack(maps).astype(np.float32)


class Forcings:
    """Builds the extra input channels for any time step of one period, on the model's device."""

    def __init__(
        self, times: np.ndarray, longitude: np.ndarray, static: np.ndarray, device: torch.device
    ):
        self.day = torch.tensor(day_fraction(times), dtype=torch.float32, device=device)
        self.year = torch.tensor(year_fraction(times), dtype=torch.float32, device=device)
        self.turn = torch.tensor(longitude / 360, dtype=torch.float32, device=device)
        self.static = torch.tensor(static, dtype=torch.float32, device=device)
        self.channels = CLOCK_CHANNELS + len(static)

    def __call__(self, index: torch.Tensor) -> torch.Tensor:
        """(batch, channels, latitude, longitude) for the time steps in index."""
        index = index.to(self.day.device)
        n, (_, height, width) = len(index), self.static.shape
        # the sun is overhead at a different UTC hour for every longitude
        sun = 2 * math.pi * (self.day[index][:, None] + self.turn[None, :])
        season = 2 * math.pi * self.year[index]
        clock = torch.stack([sun.sin(), sun.cos()], dim=1)[:, :, None, :]
        calendar = torch.stack([season.sin(), season.cos()], dim=1)[:, :, None, None]
        return torch.cat(
            [
                clock.expand(n, 2, height, width),
                calendar.expand(n, 2, height, width),
                self.static.expand(n, -1, -1, -1),
            ],
            dim=1,
        )
