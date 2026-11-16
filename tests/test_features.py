import numpy as np
import torch

from weather_forecaster.features import (
    Forcings,
    day_fraction,
    forcing_channels,
    static_maps,
    year_fraction,
)


def hours(*stamps):
    return np.array(stamps, dtype="datetime64[h]")


def test_day_fraction_counts_from_utc_midnight():
    times = hours("2001-03-04T00", "2001-03-04T06", "2001-03-04T12", "2001-03-04T18")
    np.testing.assert_allclose(day_fraction(times), [0, 0.25, 0.5, 0.75])


def test_year_fraction_accounts_for_leap_years():
    # 2001 has 365 days, so its midpoint is noon on 2 July. 2000 has 366.
    times = hours("2001-01-01T00", "2001-07-02T12", "2000-12-31T18")
    np.testing.assert_allclose(year_fraction(times), [0, 0.5, 1 - 6 / (366 * 24)])


def test_static_maps_are_standardized_and_end_with_latitude(weather):
    maps = static_maps(weather, ["land_sea_mask"])
    assert maps.shape == (3, 8, 16) and maps.dtype == np.float32
    np.testing.assert_allclose([maps[0].mean(), maps[0].std()], [0, 1], atol=1e-6)
    latitude = np.deg2rad(weather.latitude)
    np.testing.assert_allclose(maps[1][:, 0], np.sin(latitude), rtol=1e-6)
    np.testing.assert_allclose(maps[2][:, 5], np.cos(latitude), rtol=1e-6)


def test_forcings_have_one_map_per_channel_for_every_requested_time(weather):
    maps = static_maps(weather, ["land_sea_mask"])
    forcings = Forcings(weather.times, weather.longitude, maps, torch.device("cpu"))
    out = forcings(torch.tensor([0, 5, 9]))
    assert out.shape == (3, forcing_channels(1), 8, 16) and forcings.channels == forcing_channels(1)
    # the static maps are the same at every time
    assert torch.equal(out[0, 4:], torch.from_numpy(maps)) and torch.equal(out[0, 4:], out[2, 4:])


def test_the_clock_follows_the_sun_around_the_globe(weather):
    # index 1 is 06 UTC and index 2 is 12 UTC. Noon at longitude 0 at 12 UTC is the same
    # local time as noon at 90 degrees east at 06 UTC.
    forcings = Forcings(weather.times, weather.longitude, np.zeros((0, 8, 16)), torch.device("cpu"))
    out = forcings(torch.tensor([1, 2]))
    east_90 = int(np.argmin(np.abs(weather.longitude - 90)))
    assert torch.allclose(out[0, :2, :, east_90], out[1, :2, :, 0], atol=1e-6)
    # sine near 0 and cosine -1 mean half a turn through the local day, which is noon
    assert torch.allclose(out[1, :2, 0, 0], torch.tensor([0.0, -1.0]), atol=1e-6)
    # the season channels do not depend on where you are
    assert out[0, 2:4].std(dim=(-2, -1)).max() == 0
