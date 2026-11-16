import json

import numpy as np
import pytest

from weather_forecaster.era5 import LocalERA5

STEPPER_CHANNELS = ["geopotential_500", "temperature_850", "u_component_of_wind_500"]
N_TIME, N_LAT, N_LON = 160, 8, 16


@pytest.fixture
def weather(tmp_path):
    """Forty days of made-up weather for the stepping model, with one static field.

    Each channel is a wave drifting east at its own speed, at realistic magnitudes, so the
    next state really can be worked out from the current one.
    """
    longitude = np.linspace(0, 360, N_LON, endpoint=False)
    latitude = np.linspace(-78.75, 78.75, N_LAT)
    t = np.arange(N_TIME)[:, None, None]
    lon = np.deg2rad(longitude)[None, None, :]
    lat = np.deg2rad(latitude)[None, :, None]
    settings = [(54000, 3000, 0.3, 0.0), (275, 15, 0.2, 1.0), (10, 8, 0.4, 2.0)]
    for name, (mean, spread, speed, phase) in zip(STEPPER_CHANNELS, settings):
        wave = np.sin(lon - speed * t + phase) * np.cos(lat)
        np.save(tmp_path / f"{name}.npy", (mean + spread * wave).astype(np.float32))
    land = (np.sin(3 * lon[0]) * np.cos(2 * lat[0]) > 0).astype(np.float32)
    np.save(tmp_path / "static_land_sea_mask.npy", land)
    meta = {
        "time_start": "2000-01-01T00",
        "time_step_hours": 6,
        "n_time": N_TIME,
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "channels": STEPPER_CHANNELS,
        "static": ["land_sea_mask"],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return LocalERA5(tmp_path)
