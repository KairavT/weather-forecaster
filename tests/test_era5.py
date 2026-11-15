import json

import numpy as np
import pytest
import zarr

from weather_forecaster.era5 import PRESETS, Channel, LocalERA5, download

N_TIME, LEVELS, N_LON, N_LAT = 50, [500, 850], 8, 4
CHANNELS = [Channel("geopotential", 850), Channel("2m_temperature")]
# deliberately not aligned to the 10-step chunks: steps 3 to 37 of the fake remote
START, END, FIRST, COUNT = "1959-01-01T18", "1959-01-10T06", 3, 35


def expected(level_index, first=FIRST, count=COUNT):
    """Value at each (time, latitude, longitude): every index is readable from the number."""
    t, lat, lon = np.meshgrid(
        np.arange(first, first + count), np.arange(N_LAT), np.arange(N_LON), indexing="ij"
    )
    return (t * 1000 + level_index * 100 + lon * 10 + lat).astype(np.float32)


@pytest.fixture
def remote(tmp_path):
    """A small local copy of the remote layout: axes in (time, level, longitude, latitude)."""
    group = zarr.open_group(str(tmp_path / "remote.zarr"), mode="w", zarr_format=2)

    def add(name, values, dims, chunks):
        array = group.create_array(name, shape=values.shape, dtype=values.dtype, chunks=chunks)
        array[:] = values
        array.attrs["_ARRAY_DIMENSIONS"] = dims

    add("time", 6 * np.arange(N_TIME, dtype=np.int64), ["time"], (N_TIME,))
    add("level", np.array(LEVELS, dtype=np.int64), ["level"], (2,))
    add("latitude", np.linspace(-60, 60, N_LAT), ["latitude"], (N_LAT,))
    add("longitude", np.linspace(0, 315, N_LON), ["longitude"], (N_LON,))
    t, lev, lon, lat = np.meshgrid(
        np.arange(N_TIME), np.arange(2), np.arange(N_LON), np.arange(N_LAT), indexing="ij"
    )
    add(
        "geopotential",
        (t * 1000 + lev * 100 + lon * 10 + lat).astype(np.float32),
        ["time", "level", "longitude", "latitude"],
        (10, 2, N_LON, N_LAT),
    )
    add(
        "2m_temperature",
        (t[:, 0] * 1000 + lon[:, 0] * 10 + lat[:, 0]).astype(np.float32),
        ["time", "longitude", "latitude"],
        (10, N_LON, N_LAT),
    )
    add(
        "land_sea_mask",
        (lon[0, 0] * 10 + lat[0, 0]).astype(np.float32),
        ["longitude", "latitude"],
        (N_LON, N_LAT),
    )
    return group


def fetch(remote, out, **kwargs):
    download(
        CHANNELS, START, END, out, remote=remote, chunks_per_block=2, log=lambda _: None, **kwargs
    )


def test_download_cuts_the_level_and_reorders_axes(remote, tmp_path):
    fetch(remote, tmp_path / "data")
    data = LocalERA5(tmp_path / "data")
    assert data.channels == ["geopotential_850", "2m_temperature"]
    assert data.times[0] == np.datetime64(START) and data.times[-1] == np.datetime64(END)
    np.testing.assert_array_equal(data.channel("geopotential_850"), expected(level_index=1))
    np.testing.assert_array_equal(data.channel("2m_temperature"), expected(level_index=0))


def interrupt_after(blocks):
    seen = []

    def log(message):
        seen.append(message)
        if len(seen) == blocks:
            raise KeyboardInterrupt

    return log


def test_an_interrupted_download_cannot_be_read_and_resumes_where_it_stopped(remote, tmp_path):
    out = tmp_path / "data"
    with pytest.raises(KeyboardInterrupt):
        download(
            CHANNELS, START, END, out, remote=remote, chunks_per_block=2, log=interrupt_after(1)
        )
    with pytest.raises(ValueError, match="not fully downloaded"):
        LocalERA5(out).channel("geopotential_850")

    finished_before = json.loads((out / "progress.json").read_text())["geopotential_850"]
    resumed = []
    download(CHANNELS, START, END, out, remote=remote, chunks_per_block=2, log=resumed.append)
    assert finished_before == [[3, 20]]
    assert not any("geopotential: block 1/" in line for line in resumed)
    np.testing.assert_array_equal(LocalERA5(out).channel("geopotential_850"), expected(1))


def test_a_deleted_file_is_downloaded_again(remote, tmp_path):
    out = tmp_path / "data"
    fetch(remote, out)
    (out / "2m_temperature.npy").unlink()
    fetch(remote, out)
    np.testing.assert_array_equal(LocalERA5(out).channel("2m_temperature"), expected(0))


def test_resuming_with_a_different_block_size_leaves_no_gaps(remote, tmp_path):
    out = tmp_path / "data"
    with pytest.raises(KeyboardInterrupt):
        download(
            CHANNELS, START, END, out, remote=remote, chunks_per_block=1, log=interrupt_after(1)
        )
    download(CHANNELS, START, END, out, remote=remote, chunks_per_block=4, log=lambda _: None)
    np.testing.assert_array_equal(LocalERA5(out).channel("geopotential_850"), expected(1))


def test_requests_outside_the_data_are_refused(remote, tmp_path):
    with pytest.raises(ValueError, match="outside the data"):
        download(CHANNELS, START, "1990-01-01T00", tmp_path / "a", remote=remote)
    fetch(remote, tmp_path / "data")
    data = LocalERA5(tmp_path / "data")
    assert data.period(START, END) == slice(0, COUNT)
    with pytest.raises(ValueError, match="outside the data"):
        data.period(START, "1959-02-01T00")


def test_a_level_on_a_surface_variable_is_refused(remote, tmp_path):
    with pytest.raises(ValueError, match="level"):
        download([Channel("2m_temperature", 500)], START, END, tmp_path / "data", remote=remote)


def test_static_fields_are_saved_as_latitude_by_longitude(remote, tmp_path):
    fetch(remote, tmp_path / "data", static=["land_sea_mask"])
    data = LocalERA5(tmp_path / "data")
    lat, lon = np.meshgrid(np.arange(N_LAT), np.arange(N_LON), indexing="ij")
    assert data.static_fields == ["land_sea_mask"]
    np.testing.assert_array_equal(data.static("land_sea_mask"), lon * 10 + lat)
    with pytest.raises(ValueError, match="not downloaded"):
        data.static("geopotential_at_surface")


def test_a_field_that_changes_over_time_is_refused_as_static(remote, tmp_path):
    with pytest.raises(ValueError, match="changes over time"):
        fetch(remote, tmp_path / "data", static=["2m_temperature"])


def test_the_forecast_preset_has_23_distinct_channels():
    names = [channel.name for channel in PRESETS["forecast"]]
    assert len(names) == len(set(names)) == 23
    assert {"geopotential_500", "temperature_850", "2m_temperature"} <= set(names)
