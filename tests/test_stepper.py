import numpy as np
import torch
from conftest import STEPPER_CHANNELS

from weather_forecaster import metrics
from weather_forecaster.features import Forcings, forcing_channels, static_maps
from weather_forecaster.models import ResNet
from weather_forecaster.stepper import Stepper, lead_time_rmse, load_fields, rollout

TRAIN = ("2000-01-01T00", "2000-01-25T18")
TEST = ("2000-02-01T00", "2000-02-09T18")


def untrained(weather, scales):
    net = ResNet(len(STEPPER_CHANNELS) + forcing_channels(1), len(STEPPER_CHANNELS), 16, 1)
    maps = static_maps(weather, ["land_sea_mask"])
    period = weather.period(*TEST)
    forcings = Forcings(weather.times[period], weather.longitude, maps, torch.device("cpu"))
    return Stepper(net, scales), forcings


def test_fields_are_normalized_with_scales_measured_on_the_period(weather):
    fields, scales = load_fields(weather, STEPPER_CHANNELS, *TRAIN)
    assert fields.shape == (100, 3, 8, 16) and fields.dtype == np.float32
    np.testing.assert_allclose(fields.mean(axis=(0, 2, 3)), 0, atol=1e-4)
    np.testing.assert_allclose(fields.std(axis=(0, 2, 3)), 1, atol=1e-4)
    raw = np.asarray(weather.channel("temperature_850")[weather.period(*TRAIN)])
    np.testing.assert_allclose(scales.step_std[1], np.diff(raw, axis=0).std(), rtol=1e-5)


def test_other_periods_reuse_the_training_scales(weather):
    _, scales = load_fields(weather, STEPPER_CHANNELS, *TRAIN)
    fields, same = load_fields(weather, STEPPER_CHANNELS, *TEST, scales)
    raw = np.asarray(weather.channel("geopotential_500")[weather.period(*TEST)])
    assert same is scales
    np.testing.assert_allclose(fields[:, 0], (raw - scales.mean[0]) / scales.std[0], rtol=1e-5)


def test_an_untrained_stepper_leaves_the_state_unchanged(weather):
    fields, scales = load_fields(weather, STEPPER_CHANNELS, *TEST, None)
    stepper, forcings = untrained(weather, scales)
    index = torch.tensor([0, 3])
    states = list(rollout(stepper, torch.from_numpy(fields)[index], forcings, index, steps=3))
    assert len(states) == 3 and torch.equal(states[-1], torch.from_numpy(fields)[index])


def test_lead_time_scores_of_an_untrained_stepper_are_the_persistence_baseline(weather):
    # predicting no change is persistence, whose score is computed independently here
    _, scales = load_fields(weather, STEPPER_CHANNELS, *TRAIN)
    fields, _ = load_fields(weather, STEPPER_CHANNELS, *TEST, scales)
    stepper, forcings = untrained(weather, scales)
    rmse = lead_time_rmse(
        stepper, torch.from_numpy(fields), forcings, scales, weather.latitude, steps=6, batch_size=7
    )
    assert rmse.shape == (6, 3)
    for c, name in enumerate(STEPPER_CHANNELS):
        truth = np.asarray(weather.channel(name)[weather.period(*TEST)])
        for lead in (1, 4, 6):
            expected = metrics.rmse(truth[:-lead], truth[lead:], weather.latitude)
            np.testing.assert_allclose(rmse[lead - 1, c], expected, rtol=1e-4)


def test_lead_times_longer_than_the_period_have_no_score(weather):
    fields, scales = load_fields(weather, STEPPER_CHANNELS, "2000-02-01T00", "2000-02-01T18")
    stepper, forcings = untrained(weather, scales)
    rmse = lead_time_rmse(
        stepper, torch.from_numpy(fields), forcings, scales, weather.latitude, steps=5
    )
    assert np.isfinite(rmse[:3]).all() and np.isnan(rmse[3:]).all()
