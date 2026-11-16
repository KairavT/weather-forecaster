"""Score the no-model baselines and compare them with the published WeatherBench numbers.

    uv run python scripts/baselines.py

Follows the WeatherBench setup: climatologies come from 1979 to 2016, scores from 2017 to 2018.
Persistence is also scored at every 6-hour lead time out to 10 days, for the lead-time chart.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from weather_forecaster import baselines, metrics
from weather_forecaster.era5 import STEP_HOURS, LocalERA5

TRAIN = ("1979-01-01T00", "2016-12-31T18")
TEST = ("2017-01-01T00", "2018-12-31T18")
LEAD_DAYS = (3, 5)
CURVE_DAYS = 10  # persistence is scored at every step out to this lead time

# short name -> (local channel, unit, decimals to print)
TARGETS = {
    "Z500": ("geopotential_500", "m^2/s^2", 0),
    "T850": ("temperature_850", "K", 2),
    "T2M": ("2m_temperature", "K", 2),
}

# Table 2 of Rasp et al. 2020, "WeatherBench: A benchmark dataset for data-driven weather
# forecasting" (arxiv.org/abs/2002.00469)
PUBLISHED = {
    "Z500": {
        "persistence_3d": 936,
        "persistence_5d": 1033,
        "climatology": 1075,
        "weekly_climatology": 816,
    },
    "T850": {
        "persistence_3d": 4.23,
        "persistence_5d": 4.56,
        "climatology": 5.51,
        "weekly_climatology": 3.50,
    },
    "T2M": {
        "persistence_3d": 3.00,
        "persistence_5d": 3.27,
        "climatology": 6.07,
        "weekly_climatology": 3.19,
    },
}


def score_target(data: LocalERA5, channel: str) -> dict[str, float]:
    field = data.channel(channel)
    train, test = data.period(*TRAIN), data.period(*TEST)
    truth = np.asarray(field[test])
    latitude = data.latitude

    scores = {}
    for days in LEAD_DAYS:
        forecast, target = baselines.persistence(truth, days * 24 // STEP_HOURS)
        scores[f"persistence_{days}d"] = metrics.rmse(forecast, target, latitude)

    scores["climatology"] = metrics.rmse(baselines.climatology(field[train]), truth, latitude)

    by_week = baselines.weekly_climatology(field[train], data.times[train])
    weekly_forecast = by_week[baselines.iso_week(data.times[test])]
    scores["weekly_climatology"] = metrics.rmse(weekly_forecast, truth, latitude)
    return scores


def persistence_curve(data: LocalERA5, channel: str, steps: int) -> list[float]:
    truth = np.asarray(data.channel(channel)[data.period(*TEST)])
    return baselines.persistence_rmse_by_lead(truth, data.latitude, steps)


def to_markdown(results: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Field | Baseline | This repo | Published | Difference |",
        "|---|---|---|---|---|",
    ]
    for target, scores in results.items():
        _, unit, decimals = TARGETS[target]
        for name, value in scores.items():
            published = PUBLISHED.get(target, {}).get(name)
            if published is None:
                reference, difference = "n/a", "n/a"
            else:
                reference = f"{published:.{decimals}f}"
                difference = f"{100 * (value - published) / published:+.1f}%"
            label = name.replace("_", " ")
            lines.append(
                f"| {target} ({unit}) | {label} | {value:.{decimals}f} | {reference} | {difference} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score persistence and climatology baselines.")
    parser.add_argument("--data", type=Path, default=Path("data/era5_64x32"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = LocalERA5(args.data)
    results = {target: score_target(data, channel) for target, (channel, _, _) in TARGETS.items()}

    table = to_markdown(results)
    print(table)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baselines.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "baselines.md").write_text(table + "\n")

    steps = CURVE_DAYS * 24 // STEP_HOURS
    curves = {"lead_hours": [STEP_HOURS * (k + 1) for k in range(steps)]}
    for target, (channel, _, _) in TARGETS.items():
        curves[target] = persistence_curve(data, channel, steps)
    (args.out / "persistence_by_lead.json").write_text(json.dumps(curves, indent=2) + "\n")


if __name__ == "__main__":
    main()
