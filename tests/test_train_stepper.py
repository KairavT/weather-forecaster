import math

import numpy as np
import pytest
import torch
from conftest import STEPPER_CHANNELS

from weather_forecaster.train_stepper import (
    StepperConfig,
    load_stepper,
    save_stepper,
    score_lead_times,
    train_stepper,
)


def config(**overrides):
    settings = {
        "channels": STEPPER_CHANNELS,
        "static": ["land_sea_mask"],
        "width": 16,
        "blocks": 1,
        "phases": [(1, 3), (2, 1)],
        "batch_size": 16,
        "warmup_updates": 2,
        "validation_steps": 4,
        "device": "cpu",
        "train": ("2000-01-01T00", "2000-01-25T18"),
        "validation": ("2000-01-26T00", "2000-01-31T18"),
        "test": ("2000-02-01T00", "2000-02-09T18"),
    }
    return StepperConfig(**{**settings, **overrides})


def quiet(_):
    pass


def test_training_lowers_the_loss_and_moves_on_to_longer_rollouts(weather):
    result = train_stepper(config(), weather, log=quiet)
    history = result.history
    assert [epoch["rollout_steps"] for epoch in history] == [1, 1, 1, 2]
    assert history[2]["train_loss"] < history[0]["train_loss"]
    assert all(math.isfinite(epoch["validation_score"]) for epoch in history)
    assert set(history[0]["validation_rmse"]) == {"geopotential_500", "temperature_850"}


def test_every_channel_is_scored_at_every_lead_time(weather):
    settings = config(phases=[(1, 1)])
    result = train_stepper(settings, weather, log=quiet)
    scores = score_lead_times(result, settings, weather, settings.test, steps=5)
    assert list(scores) == STEPPER_CHANNELS
    assert all(len(v) == 5 and np.isfinite(v).all() and min(v) > 0 for v in scores.values())


def test_a_checkpoint_loads_with_the_safe_loader_and_steps_the_same(weather, tmp_path):
    settings = config(phases=[(1, 1)])
    result = train_stepper(settings, weather, log=quiet)
    save_stepper(result, settings, tmp_path / "stepper.pt")
    loaded, loaded_settings = load_stepper(tmp_path / "stepper.pt")

    assert loaded_settings == settings
    np.testing.assert_array_equal(loaded.scales.step_std, result.scales.step_std)
    state, forcing = torch.randn(2, 3, 8, 16), torch.randn(2, 7, 8, 16)
    with torch.no_grad():
        assert torch.equal(loaded.stepper(state, forcing), result.stepper(state, forcing))


def test_an_interrupted_run_resumes_and_ends_identical_to_an_uninterrupted_one(weather, tmp_path):
    settings = config()
    straight = train_stepper(settings, weather, log=quiet)

    def killed_during_epoch_3(message):
        if message.startswith("epoch   3"):
            raise KeyboardInterrupt

    state = tmp_path / "state.pt"
    with pytest.raises(KeyboardInterrupt):
        train_stepper(settings, weather, log=killed_during_epoch_3, state_path=state)
    messages = []
    resumed = train_stepper(settings, weather, log=messages.append, state_path=state)

    assert "resuming after epoch 2" in messages
    assert not any(message.startswith("epoch   1") for message in messages)
    assert resumed.history == straight.history
    weights = zip(resumed.stepper.state_dict().values(), straight.stepper.state_dict().values())
    assert all(torch.equal(a, b) for a, b in weights)


def test_a_time_limit_stops_early_but_still_returns_a_model(weather):
    messages = []
    result = train_stepper(config(), weather, log=messages.append, max_hours=0)
    assert len(result.history) == 1
    assert any("stopping after epoch 1" in message for message in messages)


def test_settings_that_cannot_work_are_refused_before_training_starts(weather):
    with pytest.raises(ValueError, match="phase"):
        train_stepper(config(phases=[]), weather)
    with pytest.raises(ValueError, match="phase"):
        train_stepper(config(phases=[(0, 1)]), weather)
    with pytest.raises(ValueError, match="multiple of 8"):
        train_stepper(config(width=20), weather)
    with pytest.raises(ValueError, match="too short"):
        train_stepper(config(validation_steps=24), weather)


def test_a_diverged_run_is_reported_as_such(weather):
    with pytest.raises(RuntimeError, match="diverged"):
        train_stepper(config(learning_rate=1e30, clip_norm=1e30), weather, log=quiet)
