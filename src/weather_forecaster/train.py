"""Train a network to map the weather now to the weather a fixed lead time later."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from weather_forecaster import metrics
from weather_forecaster.dataset import Normalizer, fit_normalizer, stack_channels
from weather_forecaster.era5 import LocalERA5
from weather_forecaster.models import SimpleCNN


@dataclass
class TrainConfig:
    channels: list[str]
    lead_steps: int  # how many 6-hour steps ahead the network predicts
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 5e-4
    patience: int = 5  # stop after this many epochs without a better validation loss
    seed: int = 0
    # the WeatherBench split: train on the past, tune on 2016, report on 2017 and 2018
    train: tuple[str, str] = ("1979-01-01T00", "2015-12-31T18")
    validation: tuple[str, str] = ("2016-01-01T00", "2016-12-31T18")
    test: tuple[str, str] = ("2017-01-01T00", "2018-12-31T18")


@dataclass
class TrainResult:
    model: torch.nn.Module
    normalizer: Normalizer
    history: list[dict] = field(default_factory=list)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Mean squared error with each latitude row weighted by the area it covers."""
    return (weights * (prediction - target) ** 2).mean()


@torch.no_grad()
def predict(model: torch.nn.Module, inputs: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    return torch.cat([model(inputs[i : i + batch_size]) for i in range(0, len(inputs), batch_size)])


def train(config: TrainConfig, data: LocalERA5, log=print) -> TrainResult:
    torch.manual_seed(config.seed)
    device = pick_device()

    train_fields = stack_channels(data, config.channels, *config.train)
    normalizer = fit_normalizer(train_fields)

    def on_device(fields: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(normalizer.normalize(fields)).to(device)

    # the whole training set fits in memory, so batches are cut straight from one tensor
    train_x = on_device(train_fields)
    val_x = on_device(stack_channels(data, config.channels, *config.validation))
    weights = torch.tensor(
        metrics.latitude_weights(data.latitude), dtype=torch.float32, device=device
    )[:, None]

    model = SimpleCNN(len(config.channels)).to(device)
    n_starts = len(train_x) - config.lead_steps
    steps_per_epoch = -(-n_starts // config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * steps_per_epoch
    )
    log(
        f"device {device}, {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{n_starts:,} training pairs"
    )

    result = TrainResult(model, normalizer)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(n_starts).to(device)
        running = 0.0
        for i in range(0, n_starts, config.batch_size):
            starts = order[i : i + config.batch_size]
            loss = weighted_mse(
                model(train_x[starts]), train_x[starts + config.lead_steps], weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(starts)

        val_prediction = predict(model, val_x[: -config.lead_steps])
        val_loss = weighted_mse(val_prediction, val_x[config.lead_steps :], weights).item()
        result.history.append(
            {"epoch": epoch, "train_loss": running / n_starts, "validation_loss": val_loss}
        )
        log(
            f"epoch {epoch:3d}  train {running / n_starts:.4f}  validation {val_loss:.4f}  "
            f"({time.time() - started:.0f}s)"
        )

        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                log(f"no improvement for {config.patience} epochs, stopping")
                break

    model.load_state_dict(best_state)
    return result


def score(
    result: TrainResult, config: TrainConfig, data: LocalERA5, period: tuple[str, str]
) -> dict[str, float]:
    """Benchmark RMSE of each channel in physical units over a period."""
    device = next(result.model.parameters()).device
    fields = stack_channels(data, config.channels, *period)
    inputs = torch.from_numpy(result.normalizer.normalize(fields[: -config.lead_steps])).to(device)
    prediction = result.normalizer.denormalize(predict(result.model, inputs).cpu().numpy())
    truth = fields[config.lead_steps :]
    return {
        name: metrics.rmse(prediction[:, c], truth[:, c], data.latitude)
        for c, name in enumerate(config.channels)
    }
