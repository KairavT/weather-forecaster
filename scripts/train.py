"""Train the baseline CNN for one lead time and score it on the test years.

uv run python scripts/train.py --lead-days 3
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from weather_forecaster.era5 import STEP_HOURS, LocalERA5
from weather_forecaster.train import TrainConfig, score, train

CHANNELS = ["geopotential_500", "temperature_850"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline CNN.")
    parser.add_argument("--lead-days", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--data", type=Path, default=Path("data/era5_64x32"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("results/models.json"))
    args = parser.parse_args()

    name = f"cnn-{args.lead_days}d"
    config = TrainConfig(
        channels=CHANNELS, lead_steps=args.lead_days * 24 // STEP_HOURS, epochs=args.epochs
    )
    data = LocalERA5(args.data)
    result = train(config, data)
    test_rmse = score(result, config, data, config.test)
    print(f"{name} test RMSE:", {k: round(v, 2) for k, v in test_rmse.items()})

    run_dir = args.runs / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": result.model.state_dict(),
            "mean": result.normalizer.mean,
            "std": result.normalizer.std,
            "config": asdict(config),
        },
        run_dir / "model.pt",
    )
    (run_dir / "history.json").write_text(json.dumps(result.history, indent=2) + "\n")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    results[name] = {
        "lead_days": args.lead_days,
        "epochs_run": len(result.history),
        "test_rmse": test_rmse,
    }
    args.results.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
