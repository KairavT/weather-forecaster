"""Draw one forecast from the trained stepper next to what really happened.

    uv run python scripts/download.py --preset forecast --start 2017-01-01T00 \
        --end 2018-12-31T18 --out data/era5_forecast_test
    uv run python scripts/plot_forecast.py --model runs/stepper/model.pt

The forecast shown is not chosen by eye. Forecasts are started every 3 days through the test
years, and the one whose 5-day Z500 error is the median is drawn, so it is a typical one.
"""

import argparse
from pathlib import Path

import numpy as np

from weather_forecaster import baselines, metrics
from weather_forecaster.era5 import STEP_HOURS, LocalERA5
from weather_forecaster.forecast import Forecaster, typical
from weather_forecaster.plots import forecast_maps
from weather_forecaster.train import pick_device
from weather_forecaster.train_stepper import load_stepper

CHANNEL = "geopotential_500"
LEAD_DAYS = (1, 3, 5)
EVERY = 3 * 24 // STEP_HOURS  # the typical forecast is picked from starts this many steps apart
BAND = 600  # width of one colour step on the map, in m^2/s^2
# the usual pattern for each week of the year is averaged over the same years as the baselines
USUAL = ("1979-01-01T00", "2016-12-31T18")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a forecast next to what happened.")
    parser.add_argument("--model", type=Path, required=True, help="model.pt from a stepper run")
    parser.add_argument("--data", type=Path, default=Path("data/era5_forecast_test"))
    parser.add_argument("--usual", type=Path, default=Path("data/era5_64x32"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--start", help="start time like 2018-03-14T00, instead of a typical one")
    args = parser.parse_args()

    result, config = load_stepper(args.model, device=str(pick_device()))
    data = LocalERA5(args.data)
    forecaster = Forecaster(result, config, data, config.test)
    leads = [days * 24 // STEP_HOURS for days in LEAD_DAYS]
    steps = max(leads)

    if args.start:
        found = np.flatnonzero(forecaster.times == np.datetime64(args.start, "h"))
        if len(found) == 0 or found[0] + steps >= len(forecaster.times):
            raise SystemExit(f"{args.start} is not a start time with {steps} steps after it")
        start = int(found[0])
    else:
        starts = np.arange(0, len(forecaster.times) - steps, EVERY)
        errors = forecaster.errors(starts, CHANNEL, steps)
        start = typical(starts, errors)
        print(
            f"{len(starts)} forecasts, {LEAD_DAYS[-1]}-day error from {errors.min():.0f} to "
            f"{errors.max():.0f}, median {np.median(errors):.0f}"
        )

    c = config.channels.index(CHANNEL)
    rows = [lead - 1 for lead in leads]
    forecast = forecaster.run(start, steps)[rows, c]
    truth = forecaster.truth(start, steps)[rows, c]

    climate = LocalERA5(args.usual)
    years = climate.period(*USUAL)
    by_week = baselines.weekly_climatology(climate.channel(CHANNEL)[years], climate.times[years])
    usual = by_week[baselines.iso_week(forecaster.times[[start + lead for lead in leads]])]

    started = forecaster.times[start].astype(object)
    notes = [
        f"error {metrics.rmse(forecast[k], truth[k], data.latitude):,.0f} m²/s²"
        for k in range(len(leads))
    ]
    figure = forecast_maps(
        data.longitude,
        data.latitude,
        truth - usual,
        forecast - usual,
        data.static("land_sea_mask"),
        headings=[f"After {days} day{'s' if days > 1 else ''}" for days in LEAD_DAYS],
        notes=notes,
        band=BAND,
        title=f"A typical forecast of Z500, started {started:%-d %B %Y} at {started:%H:%M} UTC",
        subtitle=(
            "Difference from the usual pattern for the week of the year. "
            "Blue is lower pressure than usual, red is higher."
        ),
        scale_label="Z500 difference from usual (m²/s²)",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "forecast_z500.png"
    figure.savefig(path)
    print(f"start {forecaster.times[start]}, {', '.join(notes)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
