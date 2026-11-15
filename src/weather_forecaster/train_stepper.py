"""Train the stepping model: single steps first, then longer and longer rollouts.

A network trained only to get the next 6 hours right falls apart when it is fed its own
output, because it has never seen its own mistakes as input. So after a first phase on single
steps, training rolls the model forward several steps and compares every one of them with what
really happened. The error at step 8 then flows back through all 8 steps, which teaches the
network to make forecasts that stay sensible when they are continued.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from weather_forecaster import metrics
from weather_forecaster.era5 import LocalERA5
from weather_forecaster.features import Forcings, forcing_channels, static_maps
from weather_forecaster.models import ResNet
from weather_forecaster.stepper import (
    Scales,
    Stepper,
    lead_time_rmse,
    load_fields,
    precision,
    rollout,
)
from weather_forecaster.train import load_state, pick_device, save_state

# the fields WeatherBench reports, logged every epoch when the model predicts them
WATCHED = ("geopotential_500", "temperature_850", "2m_temperature")


@dataclass
class StepperConfig:
    channels: list[str]
    static: list[str] = field(default_factory=list)  # fixed maps the network gets as inputs
    width: int = 128
    blocks: int = 8
    # (rollout length in steps, epochs), in the order they are trained
    phases: list[tuple[int, int]] = field(default_factory=lambda: [(1, 16), (2, 6), (4, 6), (8, 4)])
    # network steps per update. A 4-step rollout starts from a quarter as many states, so every
    # update costs the same time and memory whatever the rollout length.
    batch_size: int = 64
    learning_rate: float = 1e-3
    final_learning_rate: float = 3e-5
    warmup_updates: int = 500
    clip_norm: float = 1.0
    validation_steps: int = 12  # the best weights are picked by the forecast this far ahead
    seed: int = 0
    device: str | None = None  # None picks the fastest device available
    train: tuple[str, str] = ("1979-01-01T00", "2015-12-31T18")
    validation: tuple[str, str] = ("2016-01-01T00", "2016-12-31T18")
    test: tuple[str, str] = ("2017-01-01T00", "2018-12-31T18")


@dataclass
class StepperResult:
    stepper: Stepper
    scales: Scales
    history: list[dict] = field(default_factory=list)


def build_stepper(config: StepperConfig, scales: Scales) -> Stepper:
    n = len(config.channels)
    inputs = n + forcing_channels(len(config.static))
    return Stepper(ResNet(inputs, n, width=config.width, blocks=config.blocks), scales)


def rollout_loss(
    stepper: Stepper,
    fields: torch.Tensor,
    forcings: Forcings,
    starts: torch.Tensor,
    horizon: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Error averaged over every step of a rollout, in units of one step's typical change.

    fields stays in main memory and only the states a batch needs are moved to the device,
    which leaves the GPU's memory for the rollout itself.
    """
    device = weights.device
    total = 0.0
    states = rollout(stepper, fields[starts].to(device), forcings, starts, horizon)
    for k, state in enumerate(states, start=1):
        error = (state - fields[starts + k].to(device)) / stepper.step_scale
        total = total + (weights * error**2).mean()
    return total / horizon


def train_stepper(
    config: StepperConfig,
    data: LocalERA5,
    log=print,
    state_path: Path | None = None,
    max_hours: float | None = None,
) -> StepperResult:
    """Train through every phase and return the weights that validated best.

    With state_path, progress is saved after every epoch and a later call with the same
    settings continues from there. With max_hours, training stops early rather than start an
    epoch it could not finish in time, so a session with a time limit still ends with a model.
    """
    if not config.phases or any(horizon < 1 or epochs < 1 for horizon, epochs in config.phases):
        raise ValueError("every phase needs a rollout length and an epoch count of at least 1")
    if config.width % 8:
        raise ValueError("width must be a multiple of 8")
    run_started = time.time()
    torch.manual_seed(config.seed)
    device = torch.device(config.device) if config.device else pick_device()
    half = device.type == "cuda"

    fields, scales = load_fields(data, config.channels, *config.train)
    val_fields, _ = load_fields(data, config.channels, *config.validation, scales)
    fields, val_fields = torch.from_numpy(fields), torch.from_numpy(val_fields)
    # every start must have truth for the longest rollout of any phase
    n_starts = len(fields) - max(horizon for horizon, _ in config.phases)
    if n_starts < config.batch_size or len(val_fields) <= config.validation_steps:
        raise ValueError("the training or validation period is too short for these settings")
    maps = static_maps(data, config.static)
    forcings = Forcings(data.times[data.period(*config.train)], data.longitude, maps, device)
    val_forcings = Forcings(
        data.times[data.period(*config.validation)], data.longitude, maps, device
    )
    weights = torch.tensor(
        metrics.latitude_weights(data.latitude), dtype=torch.float32, device=device
    )[:, None]

    stepper = build_stepper(config, scales).to(device)
    horizons = [horizon for horizon, epochs in config.phases for _ in range(epochs)]
    updates_per_epoch = n_starts // config.batch_size
    total_updates = updates_per_epoch * len(horizons)
    warmup = min(config.warmup_updates, total_updates // 10)
    floor = config.final_learning_rate / config.learning_rate

    def learning_rate_factor(update: int) -> float:
        """Ramp up briefly, then glide down along a cosine to the final rate."""
        if update < warmup:
            return (update + 1) / warmup
        progress = min(1.0, (update - warmup) / max(1, total_updates - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(stepper.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    # half precision needs its gradients scaled up so that small ones do not round to zero
    scaler = torch.amp.GradScaler("cuda", enabled=half)
    log(
        f"device {device}, {sum(p.numel() for p in stepper.parameters()):,} parameters, "
        f"{len(config.channels)} channels, {n_starts:,} training starts, "
        f"{updates_per_epoch} updates per epoch"
    )

    result = StepperResult(stepper, scales)
    best_score, best_state, first_epoch = float("inf"), None, 1
    saved = load_state(state_path, config, device, log) if state_path else None
    if saved:
        stepper.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best_score, best_state = saved["best_score"], saved["best_state"]
        result.history = saved["history"]
        first_epoch = saved["epoch"] + 1
        log(f"resuming after epoch {saved['epoch']}")

    for epoch in range(first_epoch, len(horizons) + 1):
        started = time.time()
        horizon = horizons[epoch - 1]
        per_update = max(1, config.batch_size // horizon)
        stepper.train()
        # the shuffle depends only on the seed and the epoch, so a resumed run repeats it
        shuffle = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(n_starts, generator=shuffle)[: updates_per_epoch * per_update]
        running = 0.0
        for i in range(0, len(order), per_update):
            with precision(device, half):
                loss = rollout_loss(
                    stepper, fields, forcings, order[i : i + per_update], horizon, weights
                )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(stepper.parameters(), config.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
        train_loss = running / updates_per_epoch
        train_seconds = time.time() - started

        rmse = lead_time_rmse(
            stepper,
            val_fields,
            val_forcings,
            scales,
            data.latitude,
            config.validation_steps,
            stride=2,
            half_precision=half,
        )[-1]
        # one number for all channels: squared error in units of each channel's own spread
        score = float(np.mean((rmse / scales.std) ** 2))
        if not (math.isfinite(train_loss) and math.isfinite(score)):
            raise RuntimeError(f"training diverged in epoch {epoch}: loss {train_loss}")
        watched = {
            name: float(rmse[config.channels.index(name)])
            for name in WATCHED
            if name in config.channels
        }
        result.history.append(
            {
                "epoch": epoch,
                "rollout_steps": horizon,
                "train_loss": train_loss,
                "validation_score": score,
                "validation_rmse": watched,
            }
        )
        speed = updates_per_epoch * per_update * horizon / train_seconds
        log(
            f"epoch {epoch:3d}  rollout {horizon}  train {train_loss:.4f}  "
            f"validation {score:.4f}  "
            + "  ".join(f"{name} {value:.4g}" for name, value in watched.items())
            + f"  ({time.time() - started:.0f}s, {speed:.0f} steps/s)"
        )

        if score < best_score:
            best_score, best_state = score, copy.deepcopy(stepper.state_dict())
        if state_path:
            save_state(
                state_path,
                {
                    "config": asdict(config),
                    "epoch": epoch,
                    "model": stepper.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_score": best_score,
                    "best_state": best_state,
                    "history": result.history,
                },
            )
        out_of_time = (
            max_hours is not None
            and time.time() - run_started + 1.25 * (time.time() - started) > max_hours * 3600
        )
        if out_of_time and epoch < len(horizons):
            log(f"the next epoch would not fit in {max_hours} hours, stopping after epoch {epoch}")
            break

    stepper.load_state_dict(best_state)
    return result


def score_lead_times(
    result: StepperResult,
    config: StepperConfig,
    data: LocalERA5,
    period: tuple[str, str],
    steps: int,
) -> dict[str, list[float]]:
    """Benchmark RMSE of each channel at every lead time up to steps, in physical units."""
    device = result.stepper.step_scale.device
    fields, _ = load_fields(data, config.channels, *period, result.scales)
    forcings = Forcings(
        data.times[data.period(*period)],
        data.longitude,
        static_maps(data, config.static),
        device,
    )
    rmse = lead_time_rmse(
        result.stepper,
        torch.from_numpy(fields),
        forcings,
        result.scales,
        data.latitude,
        steps,
        half_precision=device.type == "cuda",
    )
    return {name: rmse[:, c].tolist() for c, name in enumerate(config.channels)}


def save_stepper(result: StepperResult, config: StepperConfig, path: Path) -> None:
    """Save the weights with the scales and settings needed to use them.

    Like the direct model's checkpoint, it holds only tensors and plain values, so it loads
    with PyTorch's safe loader.
    """
    torch.save(
        {
            "model": result.stepper.state_dict(),
            "mean": result.scales.mean.tolist(),
            "std": result.scales.std.tolist(),
            "step_std": result.scales.step_std.tolist(),
            "config": asdict(config),
        },
        path,
    )


def load_stepper(path: Path, device: str = "cpu") -> tuple[StepperResult, StepperConfig]:
    """Rebuild the stepping model from a file written by save_stepper."""
    saved = torch.load(path, map_location=device, weights_only=True)
    config = StepperConfig(**saved["config"])
    scales = Scales(
        *(np.asarray(saved[key], dtype=np.float32) for key in ("mean", "std", "step_std"))
    )
    stepper = build_stepper(config, scales).to(device)
    stepper.load_state_dict(saved["model"])
    return StepperResult(stepper, scales), config
