import numpy as np
import pytest

from weather_forecaster import metrics

LATITUDE = np.linspace(-87.1875, 87.1875, 32)
SHAPE = (10, 32, 64)


@pytest.fixture
def truth():
    return np.random.default_rng(0).normal(size=SHAPE)


def test_weights_average_one_and_favor_the_equator():
    weights = metrics.latitude_weights(LATITUDE)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[16] > weights[0]


def test_rmse_is_zero_for_a_perfect_forecast(truth):
    assert metrics.rmse(truth, truth, LATITUDE) == 0.0


def test_rmse_of_a_constant_offset_is_the_offset(truth):
    assert metrics.rmse(truth + 3.0, truth, LATITUDE) == pytest.approx(3.0)


def test_errors_near_the_pole_count_less_than_at_the_equator():
    truth = np.zeros((1, 32, 64))
    polar, equatorial = truth.copy(), truth.copy()
    polar[:, 0] = 1.0
    equatorial[:, 16] = 1.0
    assert metrics.rmse(polar, truth, LATITUDE) < metrics.rmse(equatorial, truth, LATITUDE)


def test_a_single_map_broadcasts_against_many_time_steps(truth):
    one_map = np.zeros(SHAPE[1:])
    assert metrics.rmse(one_map, truth, LATITUDE) == pytest.approx(
        metrics.rmse(np.zeros(SHAPE), truth, LATITUDE)
    )


def test_mean_rmse_never_exceeds_rmse(truth):
    growing_error = truth + np.arange(SHAPE[0])[:, None, None]
    assert metrics.mean_rmse(growing_error, truth, LATITUDE) < metrics.rmse(
        growing_error, truth, LATITUDE
    )


def test_acc_is_one_for_perfect_and_minus_one_for_inverted(truth):
    climatology = np.zeros(SHAPE[1:])
    assert metrics.acc(truth, truth, climatology, LATITUDE) == pytest.approx(1.0)
    assert metrics.acc(-truth, truth, climatology, LATITUDE) == pytest.approx(-1.0)
