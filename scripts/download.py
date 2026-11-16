"""Download ERA5 channels from the public WeatherBench 2 bucket.

    uv run python scripts/download.py --preset baselines

Safe to interrupt: finished blocks are skipped when the command is run again.
"""

import argparse
from pathlib import Path

from weather_forecaster.era5 import PRESETS, STATIC_FIELDS, download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 channels for local use.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="baselines",
        help="baselines is 3 fields (1.5 GB), forecast is 23 fields (about 12 GB)",
    )
    parser.add_argument("--start", default="1979-01-01T00", help="first time step, inclusive")
    parser.add_argument("--end", default="2020-12-31T18", help="last time step, inclusive")
    parser.add_argument("--out", type=Path, default=Path("data/era5_64x32"))
    args = parser.parse_args()
    download(
        PRESETS[args.preset],
        args.start,
        args.end,
        args.out,
        static=STATIC_FIELDS.get(args.preset, []),
    )


if __name__ == "__main__":
    main()
