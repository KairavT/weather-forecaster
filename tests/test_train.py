import json
import math

import numpy as np
import pytest
import torch

from weather_forecaster.era5 import LocalERA5
from weather_forecaster.train import (
    TrainConfig,
    load_checkpoint,
    predict,
    save_checkpoint,
    score,
    train,
)

CHANNELS = ["geopotential_500", "temperature_850"]
N_TIME, N_LAT, N_LON = 120, 8, 16


@pytest.fixture
def data(tmp_path):
    """Thirty days of made-up weather: a wave that drifts east, at realistic magnitudes."""
    longitude = np.linspace(0, 360, N_LON, endpoint=False)
    latitude = np.linspace(-78.75, 78.75, N_LAT)
    t = np.arange(N_TIME)[:, None, None]
    wave = (
        np.sin(np.deg2rad(longitude)[None, None, :] - 0.3 * t)
        * np.cos(np.deg2rad(latitude))[None, :, None]
    )
    for name, mean, spread in [("geopotential_500", 54000, 3000), ("temperature_850", 275, 15)]:
        np.save(tmp_path / f"{name}.npy", (mean + spread * wave).astype(np.float32))
    meta = {
        "time_start": "2000-01-01T00",
        "time_step_hours": 6,
        "n_time": N_TIME,
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "channels": CHANNELS,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return LocalERA5(tmp_path)


def config(**overrides):
    settings = {
        "channels": CHANNELS,
        "lead_steps": 2,
        "epochs": 3,
        "batch_size": 16,
        "device": "cpu",
        "train": ("2000-01-01T00", "2000-01-20T18"),
        "validation": ("2000-01-21T00", "2000-01-25T18"),
        "test": ("2000-01-26T00", "2000-01-30T18"),
    }
    return TrainConfig(**{**settings, **overrides})


def test_training_lowers_the_loss_and_scores_every_channel(data):
    result = train(config(), data, log=lambda _: None)
    losses = [epoch["train_loss"] for epoch in result.history]
    assert len(losses) == 3 and losses[-1] < losses[0]
    rmse = score(result, config(), data, config().test)
    assert list(rmse) == CHANNELS and all(math.isfinite(v) and v > 0 for v in rmse.values())


def test_a_checkpoint_loads_with_the_safe_loader_and_predicts_the_same(data, tmp_path):
    settings = config(epochs=1)
    result = train(settings, data, log=lambda _: None)
    save_checkpoint(result, settings, tmp_path / "model.pt")
    loaded, loaded_settings = load_checkpoint(tmp_path / "model.pt")

    assert loaded_settings == settings
    np.testing.assert_array_equal(loaded.normalizer.mean, result.normalizer.mean)
    inputs = torch.randn(4, 2, N_LAT, N_LON)
    assert torch.equal(predict(loaded.model, inputs), predict(result.model, inputs))


def test_a_lead_time_of_zero_is_refused_before_training_starts(data):
    with pytest.raises(ValueError, match="lead_steps"):
        train(config(lead_steps=0), data)


def test_periods_shorter_than_the_lead_time_are_refused(data):
    with pytest.raises(ValueError, match="longer than the lead time"):
        train(config(lead_steps=30), data)


def test_a_diverged_run_is_reported_as_such(data):
    with pytest.raises(RuntimeError, match="diverged"):
        train(config(learning_rate=1e30), data, log=lambda _: None)


def test_an_interrupted_run_resumes_and_ends_identical_to_an_uninterrupted_one(data, tmp_path):
    settings = config(epochs=4)
    straight = train(settings, data, log=lambda _: None)

    def killed_during_epoch_3(message):
        if message.startswith("epoch   3"):
            raise KeyboardInterrupt

    state = tmp_path / "state.pt"
    with pytest.raises(KeyboardInterrupt):
        train(settings, data, log=killed_during_epoch_3, state_path=state)
    messages = []
    resumed = train(settings, data, log=messages.append, state_path=state)

    assert "resuming after epoch 2" in messages
    assert not any(message.startswith("epoch   1") for message in messages)
    assert resumed.history == straight.history
    weights = zip(resumed.model.state_dict().values(), straight.model.state_dict().values())
    assert all(torch.equal(a, b) for a, b in weights)


def test_a_finished_run_is_not_trained_again(data, tmp_path):
    settings = config(epochs=2)
    first = train(settings, data, log=lambda _: None, state_path=tmp_path / "state.pt")
    messages = []
    again = train(settings, data, log=messages.append, state_path=tmp_path / "state.pt")
    assert not any(message.startswith("epoch") for message in messages)
    assert again.history == first.history


def test_saved_progress_from_other_settings_is_refused(data, tmp_path):
    train(config(epochs=1), data, log=lambda _: None, state_path=tmp_path / "state.pt")
    with pytest.raises(ValueError, match="different settings"):
        train(config(epochs=2), data, state_path=tmp_path / "state.pt")


def test_unreadable_saved_progress_starts_over(data, tmp_path):
    (tmp_path / "state.pt").write_bytes(b"cut off mid-write")
    messages = []
    result = train(config(epochs=1), data, log=messages.append, state_path=tmp_path / "state.pt")
    assert any("starting over" in message for message in messages)
    assert len(result.history) == 1
