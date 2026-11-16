import numpy as np
import pytest
from conftest import STEPPER_CHANNELS

from weather_forecaster import metrics
from weather_forecaster.forecast import Forecaster, typical
from weather_forecaster.stepper import load_fields
from weather_forecaster.train_stepper import StepperConfig, StepperResult, build_stepper

TRAIN = ("2000-01-01T00", "2000-01-25T18")
TEST = ("2000-02-01T00", "2000-02-09T18")


@pytest.fixture
def forecaster(weather):
    """An untrained stepper, which predicts no change, so every forecast is persistence."""
    config = StepperConfig(STEPPER_CHANNELS, static=["land_sea_mask"], width=16, blocks=1)
    _, scales = load_fields(weather, STEPPER_CHANNELS, *TRAIN)
    result = StepperResult(build_stepper(config, scales), scales)
    return Forecaster(result, config, weather, TEST)


def raw(weather, name):
    return np.asarray(weather.channel(name)[weather.period(*TEST)])


def test_a_forecast_comes_back_in_physical_units(weather, forecaster):
    forecast = forecaster.run(start=5, steps=3)
    assert forecast.shape == (3, 3, 8, 16)
    for c, name in enumerate(STEPPER_CHANNELS):
        for step in range(3):  # no change from the state it started from
            np.testing.assert_allclose(forecast[step, c], raw(weather, name)[5], rtol=1e-5)


def test_truth_lines_up_with_the_forecast_times(weather, forecaster):
    truth = forecaster.truth(start=5, steps=3)
    for c, name in enumerate(STEPPER_CHANNELS):
        np.testing.assert_allclose(truth[:, c], raw(weather, name)[6:9], rtol=1e-5)


def test_each_start_gets_its_own_error(weather, forecaster):
    starts = np.array([0, 7, 20])
    errors = forecaster.errors(starts, "temperature_850", lead_steps=4, batch_size=2)
    field = raw(weather, "temperature_850")
    expected = [metrics.rmse(field[s], field[s + 4], weather.latitude) for s in starts]
    np.testing.assert_allclose(errors, expected, rtol=1e-4)


def test_the_typical_start_has_the_median_error():
    assert typical(np.array([10, 20, 30, 40, 50]), np.array([3.0, 9.0, 1.0, 5.0, 7.0])) == 40
