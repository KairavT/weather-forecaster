"""Train a network to map the weather now to the weather a fixed lead time later."""

from __future__ import annotations

import copy
import math
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    device: str | None = None  # None picks the fastest device available
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


def load_state(path: Path, config, device: torch.device, log) -> dict | None:
    """Read progress saved by an earlier, interrupted training run with the same settings."""
    if not path.exists():
        return None
    try:
        saved = torch.load(path, map_location=device, weights_only=True)
    # a file cut off while it was being written fails in one of these ways
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
        log(f"could not read saved progress ({error}), starting over")
        return None
    if saved["config"] != asdict(config):
        raise ValueError(f"{path} was saved with different settings; delete it or use a new folder")
    return saved


def save_state(path: Path, state: dict) -> None:
    """Write atomically, so being killed mid-save never leaves a broken file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(
    config: TrainConfig, data: LocalERA5, log=print, state_path: Path | None = None
) -> TrainResult:
    """Train until the epoch limit, or until the validation loss stops improving.

    With state_path, progress is saved there after every epoch, and a later call with the
    same settings picks up after the last finished epoch instead of starting over. The
    resumed run ends with exactly the weights an uninterrupted run would have.
    """
    if config.lead_steps < 1:
        raise ValueError("lead_steps must be at least 1")
    torch.manual_seed(config.seed)
    device = torch.device(config.device) if config.device else pick_device()

    train_fields = stack_channels(data, config.channels, *config.train)
    normalizer = fit_normalizer(train_fields)

    def on_device(fields: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(normalizer.normalize(fields)).to(device)

    # the whole training set fits in memory, so batches are cut straight from one tensor
    train_x = on_device(train_fields)
    del train_fields  # the raw copy is no longer needed and is as large as train_x
    val_x = on_device(stack_channels(data, config.channels, *config.validation))
    if min(len(train_x), len(val_x)) <= config.lead_steps:
        raise ValueError("the training and validation periods must be longer than the lead time")
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
    best_loss, best_state, stale, first_epoch = float("inf"), None, 0, 1
    saved = load_state(state_path, config, device, log) if state_path else None
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_loss, best_state, stale = saved["best_loss"], saved["best_state"], saved["stale"]
        result.history = saved["history"]
        first_epoch = saved["epoch"] + 1
        log(f"resuming after epoch {saved['epoch']}")

    for epoch in range(first_epoch, config.epochs + 1):
        if stale >= config.patience:  # the saved run had already stopped early
            break
        started = time.time()
        model.train()
        # the shuffle depends only on the seed and the epoch, so a resumed run repeats it
        shuffle = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(n_starts, generator=shuffle).to(device)
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
        if not math.isfinite(val_loss):
            raise RuntimeError(f"training diverged: validation loss is {val_loss} in epoch {epoch}")
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
        if state_path:
            save_state(
                state_path,
                {
                    "config": asdict(config),
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "stale": stale,
                    "history": result.history,
                },
            )
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


def save_checkpoint(result: TrainResult, config: TrainConfig, path: Path) -> None:
    """Save the weights with everything needed to use them: normalization and settings.

    Only tensors and plain Python values go in, so the file loads with PyTorch's safe
    loader, which refuses anything that could run code.
    """
    torch.save(
        {
            "model": result.model.state_dict(),
            "mean": result.normalizer.mean.tolist(),
            "std": result.normalizer.std.tolist(),
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[TrainResult, TrainConfig]:
    """Rebuild the model and its normalizer from a file written by save_checkpoint."""
    saved = torch.load(path, map_location=device, weights_only=True)
    config = TrainConfig(**saved["config"])
    model = SimpleCNN(len(config.channels)).to(device)
    model.load_state_dict(saved["model"])
    normalizer = Normalizer(
        np.asarray(saved["mean"], dtype=np.float32), np.asarray(saved["std"], dtype=np.float32)
    )
    return TrainResult(model, normalizer), config
