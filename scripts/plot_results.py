"""Draw the figures in the README from the scores in results/.

    uv run python scripts/baselines.py
    uv run python scripts/plot_results.py

The first command writes the persistence scores that the lead-time chart is drawn from.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from weather_forecaster.plots import Panel, lead_time_chart, rollout_gain_chart

# short name -> (channel, panel title)
FIELDS = {
    "Z500": ("geopotential_500", "Z500 (m²/s²)"),
    "T850": ("temperature_850", "T850 (K)"),
    "T2M": ("2m_temperature", "T2M (K)"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the result figures.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    runs = load(args.results / "stepper.json")
    stepper = runs["stepper"]
    persistence = load(args.results / "persistence_by_lead.json")
    baselines = load(args.results / "baselines.json")
    networks = load(args.results / "models.json")
    if persistence["lead_hours"] != stepper["lead_hours"]:
        raise SystemExit("the stepper and persistence are scored at different lead times")

    panels = []
    for short, (channel, title) in FIELDS.items():
        cnn = {
            entry["lead_days"]: entry["test_rmse"][channel]
            for entry in networks.values()
            if channel in entry["test_rmse"]
        }
        panels.append(
            Panel(
                title,
                stepper["test_rmse"][channel],
                persistence[short],
                baselines[short]["weekly_climatology"],
                cnn,
            )
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "lead_time.png"
    lead_time_chart(stepper["lead_hours"], panels).savefig(path)
    print(f"wrote {path}")

    # drawn once the same network has also been trained without rollouts
    without = runs.get("stepper-no-rollout")
    if without:
        if without["lead_hours"] != stepper["lead_hours"]:
            raise SystemExit("the two stepper runs are scored at different lead times")
        gains = {}
        for short, (channel, _) in FIELDS.items():
            ratio = np.array(stepper["test_rmse"][channel]) / without["test_rmse"][channel]
            gains[short] = (100 * (1 - ratio)).tolist()
        path = args.out / "rollout_gain.png"
        rollout_gain_chart(stepper["lead_hours"], gains).savefig(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
