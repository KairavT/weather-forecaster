"""ERA5 reanalysis data: download from the public WeatherBench 2 bucket, read locally.

The remote store is a Zarr dataset on a 64 x 32 longitude-latitude grid (5.625 degrees),
sampled every 6 hours from 1959 to early 2023. A "channel" is either a surface variable or
one pressure level of an atmospheric variable. Each channel is saved locally as its own
float32 .npy file with shape (time, latitude, longitude), so later code can memory-map it
and read any time step without loading the whole file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import certifi
import numpy as np
import zarr

REMOTE_URL = (
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
EPOCH = np.datetime64("1959-01-01T00", "h")  # the remote time axis counts hours from here
STEP_HOURS = 6


@dataclass(frozen=True)
class Channel:
    """One 2D field per time step: a surface variable, or an atmospheric variable at one level."""

    variable: str
    level: int | None = None  # pressure level in hPa, None for surface variables

    @property
    def name(self) -> str:
        return self.variable if self.level is None else f"{self.variable}_{self.level}"


PRESETS: dict[str, list[Channel]] = {
    # the three fields WeatherBench scores: 500 hPa geopotential, 850 hPa and 2 m temperature
    "baselines": [
        Channel("geopotential", 500),
        Channel("temperature", 850),
        Channel("2m_temperature"),
    ],
}


def open_remote(url: str = REMOTE_URL) -> zarr.Group:
    """Open the public bucket anonymously. Nothing is downloaded until arrays are sliced."""
    # Some Python builds on macOS ship without root certificates, which makes every HTTPS
    # request fail. Point them at certifi's bundle unless the user already chose one.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    return zarr.open_group(url, mode="r", storage_options={"token": "anon"})


def remote_times(remote: zarr.Group) -> np.ndarray:
    """The remote time axis as datetime64[h]."""
    return EPOCH + remote["time"][:].astype("timedelta64[h]")


def time_range(times: np.ndarray, start: str, end: str) -> tuple[int, int]:
    """Half-open index range [i0, i1) covering start through end, both inclusive."""
    i0 = int(np.searchsorted(times, np.datetime64(start, "h"), side="left"))
    i1 = int(np.searchsorted(times, np.datetime64(end, "h"), side="right"))
    if i1 <= i0:
        raise ValueError(f"no data between {start} and {end}")
    return i0, i1


def _to_time_lat_lon(field: np.ndarray, dims: list[str]) -> np.ndarray:
    """Reorder axes to (time, latitude, longitude), whatever order the remote store uses."""
    return np.transpose(field, [dims.index(d) for d in ("time", "latitude", "longitude")])


def _write_json(path: Path, payload: dict) -> None:
    """Write atomically, so an interrupted run never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def download(
    channels: list[Channel],
    start: str,
    end: str,
    out_dir: Path,
    *,
    url: str = REMOTE_URL,
    chunks_per_block: int = 8,
    log=print,
) -> None:
    """Download channels for start..end into out_dir. Safe to interrupt and run again.

    The remote arrays are chunked along time, and a pressure-level variable keeps all of its
    levels in the same chunk. So each variable is read once per block of chunks and every
    requested level is cut out of that one read. Finished blocks are recorded in
    progress.json and skipped on the next run.
    """
    remote = open_remote(url)
    times = remote_times(remote)
    i0, i1 = time_range(times, start, end)
    n_time = i1 - i0
    if not np.all(np.diff(times[i0:i1]) == np.timedelta64(STEP_HOURS, "h")):
        raise ValueError("remote time axis is not evenly spaced")

    latitude = remote["latitude"][:]
    longitude = remote["longitude"][:]
    levels = [int(v) for v in remote["level"][:]]
    shape = (n_time, len(latitude), len(longitude))

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    progress_path = out_dir / "progress.json"

    meta = {
        "source": url,
        "time_start": str(times[i0]),
        "time_step_hours": STEP_HOURS,
        "n_time": n_time,
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "channels": [],
    }
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        same_span = all(existing[k] == meta[k] for k in ("source", "time_start", "n_time"))
        if not same_span:
            raise ValueError(f"{out_dir} holds a different time range; use a new directory")
        meta["channels"] = existing["channels"]
    progress: dict[str, list[int]] = (
        json.loads(progress_path.read_text()) if progress_path.exists() else {}
    )

    by_variable: dict[str, list[Channel]] = {}
    for channel in channels:
        by_variable.setdefault(channel.variable, []).append(channel)

    for variable, group in by_variable.items():
        array = remote[variable]
        dims = list(array.attrs["_ARRAY_DIMENSIONS"])
        has_levels = "level" in dims
        for channel in group:
            if has_levels != (channel.level is not None):
                raise ValueError(f"{channel.name}: level given for a surface variable, or missing")
        field_dims = [d for d in dims if d != "level"]

        outputs = {}
        for channel in group:
            path = out_dir / f"{channel.name}.npy"
            mode = "r+" if path.exists() else "w+"
            outputs[channel.name] = np.lib.format.open_memmap(
                path, mode=mode, dtype=np.float32, shape=shape if mode == "w+" else None
            )
            if outputs[channel.name].shape != shape:
                raise ValueError(f"{path} has an unexpected shape")
            progress.setdefault(channel.name, [])
            if channel.name not in meta["channels"]:
                meta["channels"].append(channel.name)
        _write_json(meta_path, meta)

        chunk = array.chunks[dims.index("time")]
        block = chunk * chunks_per_block
        block_starts = range((i0 // chunk) * chunk, i1, block)
        for n, block_start in enumerate(block_starts, start=1):
            lo, hi = max(block_start, i0), min(block_start + block, i1)
            if all(lo in progress[channel.name] for channel in group):
                continue
            data = array[lo:hi]
            for channel in group:
                field = data
                if has_levels:
                    field = np.take(data, levels.index(channel.level), axis=dims.index("level"))
                if np.isnan(field).any():
                    raise ValueError(f"{channel.name} has missing values in steps {lo}..{hi}")
                outputs[channel.name][lo - i0 : hi - i0] = _to_time_lat_lon(field, field_dims)
            for channel in group:
                outputs[channel.name].flush()
                progress[channel.name].append(lo)
            _write_json(progress_path, progress)
            log(f"{variable}: block {n}/{len(block_starts)} ({times[lo]} to {times[hi - 1]})")


class LocalERA5:
    """Read-only view of a directory written by download()."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text())
        self.latitude = np.asarray(meta["latitude"])
        self.longitude = np.asarray(meta["longitude"])
        self.channels: list[str] = meta["channels"]
        step = np.timedelta64(meta["time_step_hours"], "h")
        self.times = np.datetime64(meta["time_start"], "h") + step * np.arange(meta["n_time"])

    def channel(self, name: str) -> np.ndarray:
        """Memory-mapped (time, latitude, longitude) array for one channel."""
        return np.load(self.root / f"{name}.npy", mmap_mode="r")

    def period(self, start: str, end: str) -> slice:
        """Time slice covering start through end, both inclusive."""
        return slice(*time_range(self.times, start, end))
