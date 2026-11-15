"""A forecast made of small steps.

The stepping model predicts how the atmosphere changes over one 6-hour step. Feeding its own
output back in gives a forecast for any lead time from a single network: 12 steps for 3 days,
40 steps for 10 days.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from weather_forecaster import metrics
from weather_forecaster.era5 import LocalERA5
from weather_forecaster.features import Forcings


@dataclass(frozen=True)
class Scales:
    """Per-channel statistics, measured on the training period only.

    mean and std put every channel on the same footing as an input. step_std is how much a
    channel typically changes over one step. The network's output is measured in those units,
    so a slow field like geopotential and a fast one like wind are equally hard targets.
    """

    mean: np.ndarray  # (channel,)
    std: np.ndarray  # (channel,)
    step_std: np.ndarray  # (channel,)


def load_fields(
    data: LocalERA5, channels: list[str], start: str, end: str, scales: Scales | None = None
) -> tuple[np.ndarray, Scales]:
    """Normalized float32 fields (time, channel, latitude, longitude) for a period.

    Without scales, they are measured on this period and returned, which is what the training
    period does. Other periods pass the training scales in. One channel is read from disk at a
    time, so the peak memory is the result plus a single channel.
    """
    period = data.period(start, end)
    shape = (period.stop - period.start, len(channels), len(data.latitude), len(data.longitude))
    fields = np.empty(shape, dtype=np.float32)
    measured = []
    for c, name in enumerate(channels):
        raw = np.asarray(data.channel(name)[period])
        if scales is None:
            mean, std = raw.mean(dtype=np.float64), raw.std(dtype=np.float64)
            measured.append((mean, std, np.diff(raw, axis=0).std(dtype=np.float64)))
        else:
            mean, std = scales.mean[c], scales.std[c]
        np.subtract(raw, mean, out=fields[:, c])
        fields[:, c] /= std
    if scales is None:
        mean, std, step_std = (np.array(column, dtype=np.float32) for column in zip(*measured))
        scales = Scales(mean, std, step_std)
    return fields, scales


class Stepper(nn.Module):
    """One step forward in time, in normalized units: state and forcings in, next state out."""

    def __init__(self, net: nn.Module, scales: Scales):
        super().__init__()
        self.net = net
        # turns the network's output, in units of a typical step, into normalized state units
        step_scale = torch.tensor(scales.step_std / scales.std, dtype=torch.float32)
        self.register_buffer("step_scale", step_scale[:, None, None])

    def forward(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        change = self.net(torch.cat([state, forcing], dim=1))
        # the sum stays in full precision even when the network runs in half precision
        return state + self.step_scale * change.float()


def precision(device: torch.device, half: bool):
    """Context that runs the network in half precision, which is about twice as fast on a GPU."""
    return torch.autocast(device.type, dtype=torch.float16) if half else contextlib.nullcontext()


def rollout(
    stepper: Stepper, state: torch.Tensor, forcings: Forcings, index: torch.Tensor, steps: int
) -> Iterator[torch.Tensor]:
    """Step forward from the states at time steps index, yielding the state after each step."""
    for k in range(steps):
        state = stepper(state, forcings(index + k))
        yield state


@torch.no_grad()
def lead_time_rmse(
    stepper: Stepper,
    fields: torch.Tensor,
    forcings: Forcings,
    scales: Scales,
    latitude: np.ndarray,
    steps: int,
    *,
    stride: int = 1,
    batch_size: int = 256,
    half_precision: bool = False,
) -> np.ndarray:
    """Benchmark RMSE in physical units for every lead time, shape (steps, channel).

    A forecast is started from every stride-th time step of fields and compared with what
    really happened. Row k holds the score k + 1 steps ahead, over every start that still has
    truth that far ahead. That is the same set of forecasts the direct models and the
    baselines are scored on, so the numbers are comparable.
    """
    stepper.eval()
    device = stepper.step_scale.device
    weights = torch.tensor(metrics.latitude_weights(latitude), dtype=torch.float32, device=device)[
        :, None
    ]
    squared = torch.zeros(steps, fields.shape[1], dtype=torch.float64)
    counts = torch.zeros(steps, dtype=torch.float64)
    starts = torch.arange(0, len(fields) - 1, stride)
    for i in range(0, len(starts), batch_size):
        index = starts[i : i + batch_size]
        state = fields[index].to(device)
        for k in range(steps):
            alive = index + k + 1 < len(fields)  # starts that still have truth this far ahead
            if not alive.any():
                break
            index, state = index[alive], state[alive.to(device)]
            with precision(device, half_precision):
                state = stepper(state, forcings(index + k))
            error = state - fields[index + k + 1].to(device)
            per_forecast = (weights * error**2).mean(dim=(-2, -1))
            squared[k] += per_forecast.sum(dim=0).cpu().double()
            counts[k] += len(index)
    # a lead time longer than the period has no forecasts and comes out as nan
    rmse = torch.sqrt(squared / counts[:, None])
    return rmse.numpy() * scales.std.astype(np.float64)
