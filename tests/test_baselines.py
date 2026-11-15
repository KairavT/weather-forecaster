import numpy as np
import pytest

from weather_forecaster import baselines


def test_persistence_pairs_each_start_with_the_field_lead_steps_later():
    truth = np.arange(10.0)[:, None, None] * np.ones((1, 4, 8))
    forecast, target = baselines.persistence(truth, 3)
    assert forecast.shape == target.shape == (7, 4, 8)
    assert np.all(target - forecast == 3.0)


def test_persistence_rejects_a_zero_lead():
    with pytest.raises(ValueError):
        baselines.persistence(np.zeros((5, 4, 8)), 0)


def test_climatology_is_the_time_mean():
    train = np.arange(6.0)[:, None, None] * np.ones((1, 4, 8))
    assert np.all(baselines.climatology(train) == 2.5)


def test_iso_week_follows_the_calendar_across_year_boundaries():
    times = np.array(
        ["2017-01-01T00", "2017-01-02T00", "2018-12-31T18", "2015-12-31T00"],
        dtype="datetime64[h]",
    )
    # Sunday 1 Jan 2017 still belongs to the last week of 2016; 2015 has 53 weeks
    assert baselines.iso_week(times).tolist() == [52, 1, 1, 53]


def test_weekly_climatology_averages_only_within_each_week():
    times = np.datetime64("2015-01-01T00", "h") + np.timedelta64(6, "h") * np.arange(4 * 365 * 2)
    weeks = baselines.iso_week(times)
    train = weeks[:, None, None] * np.ones((1, 4, 8))
    by_week = baselines.weekly_climatology(train, times)
    assert by_week.shape == (54, 4, 8)
    for week in range(1, 54):
        assert np.all(by_week[week] == week)


def test_week_53_reuses_week_52_when_the_training_years_have_none():
    # 2017 has only 52 ISO weeks
    times = np.datetime64("2017-01-02T00", "h") + np.timedelta64(6, "h") * np.arange(4 * 364)
    train = baselines.iso_week(times)[:, None, None] * np.ones((1, 4, 8))
    by_week = baselines.weekly_climatology(train, times)
    assert np.all(by_week[53] == 52)
    assert not np.isnan(by_week).any()


def test_weekly_climatology_needs_a_full_year_of_training_data():
    times = np.datetime64("2017-01-02T00", "h") + np.timedelta64(6, "h") * np.arange(40)
    with pytest.raises(ValueError, match="every calendar week"):
        baselines.weekly_climatology(np.ones((40, 4, 8)), times)
