"""Train the stepping model and score it at every lead time out to 10 days.

    uv run python scripts/download.py --preset forecast --out data/era5_forecast
    uv run python scripts/train_stepper.py

This is hours of GPU time, so it is meant for a cloud GPU (see notebooks/kaggle_train.ipynb).
Add --quick to check that everything runs: it trains for a few minutes on two years of data,
which only needs 2014 to 2017 downloaded.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from weather_forecaster.era5 import PRESETS, STATIC_FIELDS, STEP_HOURS, LocalERA5
from weather_forecaster.train_stepper import (
    StepperConfig,
    save_stepper,
    score_lead_times,
    train_stepper,
)

# short name -> (channel, decimals to print) for the fields WeatherBench reports
HEADLINE = {
    "Z500": ("geopotential_500", 0),
    "T850": ("temperature_850", 2),
    "T2M": ("2m_temperature", 2),
}
PRINT_DAYS = (1, 3, 5, 7, 10)


def parse_phases(text: str) -> list[tuple[int, int]]:
    """Turn "1:16,2:6" into [(1, 16), (2, 6)]: rollout length in steps, then epochs."""
    try:
        return [tuple(int(part) for part in phase.split(":")) for phase in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected something like 1:16,2:6, got {text}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stepping model.")
    parser.add_argument("--data", type=Path, default=Path("data/era5_forecast"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results/stepper.json"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument(
        "--phases",
        type=parse_phases,
        default=[(1, 16), (2, 6), (4, 6), (8, 4)],
        help="rollout steps and epochs per phase, like 1:16,2:6,4:6,8:4",
    )
    parser.add_argument("--lead-days", type=int, default=10, help="score out to this lead time")
    parser.add_argument("--max-hours", type=float, help="stop training early to fit this budget")
    parser.add_argument("--quick", action="store_true", help="a few minutes on two years of data")
    args = parser.parse_args()

    settings = {
        "channels": [channel.name for channel in PRESETS["forecast"]],
        "static": STATIC_FIELDS["forecast"],
        "width": args.width,
        "blocks": args.blocks,
        "phases": args.phases,
    }
    if args.quick:
        settings |= {
            "phases": [(1, 2), (2, 1), (4, 1), (8, 1)],
            "train": ("2014-01-01T00", "2015-12-31T18"),
            "test": ("2017-01-01T00", "2017-12-31T18"),
        }
    name = "stepper-quick" if args.quick else "stepper"
    config = StepperConfig(**settings)
    data = LocalERA5(args.data)
    run_dir = args.runs / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # progress is saved here after every epoch, so an interrupted run resumes when rerun
    state_path = run_dir / "state.pt"

    started = time.time()
    result = train_stepper(config, data, state_path=state_path, max_hours=args.max_hours)
    train_seconds = time.time() - started
    steps = args.lead_days * 24 // STEP_HOURS
    test_rmse = score_lead_times(result, config, data, config.test, steps)

    print(f"\n{name} test RMSE by lead time")
    print("lead   " + "  ".join(f"{short:>7}" for short in HEADLINE))
    for days in PRINT_DAYS:
        if days <= args.lead_days:
            row = (
                f"{test_rmse[channel][days * 24 // STEP_HOURS - 1]:7.{decimals}f}"
                for channel, decimals in HEADLINE.values()
            )
            print(f"{days:2d} d   " + "  ".join(row))

    save_stepper(result, config, run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(result.history, indent=2) + "\n")

    on_gpu = torch.cuda.is_available()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    results[name] = {
        "width": config.width,
        "blocks": config.blocks,
        "phases": config.phases,
        "parameters": sum(p.numel() for p in result.stepper.parameters()),
        "epochs_run": len(result.history),
        "train_seconds": round(train_seconds),
        "device": torch.cuda.get_device_name(0) if on_gpu else "not a cuda gpu",
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if on_gpu else None,
        "lead_hours": [STEP_HOURS * (k + 1) for k in range(steps)],
        "test_rmse": test_rmse,
        "history": result.history,
    }
    args.results.write_text(json.dumps(results, indent=2) + "\n")
    state_path.unlink(missing_ok=True)  # finished, so there is nothing left to resume


if __name__ == "__main__":
    main()
