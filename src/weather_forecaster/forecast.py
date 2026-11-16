"""Use a trained stepper as a forecaster: one forecast from one date, in physical units.

Scoring averages over thousands of forecasts. This module is for looking at single ones, and
for picking which one to look at without choosing by eye.
"""

from __future__ import annotations

import numpy as np
import torch

from weather_forecaster import metrics
from weather_forecaster.era5 import LocalERA5
from weather_forecaster.features import Forcings, static_maps
from weather_forecaster.stepper import load_fields, rollout
from weather_forecaster.train_stepper import StepperConfig, StepperResult


class Forecaster:
    """A trained stepper with one period of data loaded, ready to forecast from any time in it."""

    def __init__(
        self, result: StepperResult, config: StepperConfig, data: LocalERA5, period: tuple[str, str]
    ):
        self.stepper = result.stepper.eval()
        self.channels = config.channels
        self.latitude = data.latitude
        self.device = self.stepper.step_scale.device
        self.times = data.times[data.period(*period)]
        fields, _ = load_fields(data, config.channels, *period, result.scales)
        self.fields = torch.from_numpy(fields)
        self.forcings = Forcings(
            self.times, data.longitude, static_maps(data, config.static), self.device
        )
        self.mean = result.scales.mean[:, None, None]
        self.std = result.scales.std[:, None, None]

    def _physical(self, normalized: torch.Tensor) -> np.ndarray:
        return normalized.cpu().numpy() * self.std + self.mean

    @torch.no_grad()
    def run(self, start: int, steps: int) -> np.ndarray:
        """The forecast from time step start: (steps, channel, latitude, longitude)."""
        index = torch.tensor([start])
        states = rollout(
            self.stepper, self.fields[index].to(self.device), self.forcings, index, steps
        )
        return np.stack([self._physical(state[0]) for state in states])

    def truth(self, start: int, steps: int) -> np.ndarray:
        """What really happened at the times run() forecasts, in the same shape."""
        return self._physical(self.fields[start + 1 : start + steps + 1])

    @torch.no_grad()
    def errors(
        self, starts: np.ndarray, channel: str, lead_steps: int, batch_size: int = 64
    ) -> np.ndarray:
        """RMSE of one channel, lead_steps ahead, separately for the forecast from each start."""
        c = self.channels.index(channel)
        weights = torch.tensor(
            metrics.latitude_weights(self.latitude), dtype=torch.float32, device=self.device
        )[:, None]
        starts = torch.as_tensor(starts)
        scores = []
        for i in range(0, len(starts), batch_size):
            index = starts[i : i + batch_size]
            first = self.fields[index].to(self.device)
            for state in rollout(self.stepper, first, self.forcings, index, lead_steps):
                pass  # only the last step is scored
            error = state[:, c] - self.fields[index + lead_steps, c].to(self.device)
            scores.append((weights * error**2).mean(dim=(-2, -1)).sqrt().cpu())
        return torch.cat(scores).numpy() * float(self.std[c, 0, 0])


def typical(starts: np.ndarray, errors: np.ndarray) -> int:
    """The start whose error is the median, so the forecast shown is neither good nor bad."""
    return int(starts[np.argsort(errors)[len(errors) // 2]])
