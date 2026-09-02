"""One-command local baseline generation and evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "res/provided_res/baseline"))

from baselines import make_all_baselines  # noqa: E402
from src.evaluate_predictions import evaluate_file  # noqa: E402


def main() -> None:
    dataset_dir = ROOT / "res/dataset"
    required = [
        dataset_dir / "local_train.csv",
        dataset_dir / "local_validation_targets.csv",
    ]
    if not all(path.exists() for path in required):
        subprocess.run([sys.executable, str(ROOT / "src/split_data.py")], check=True)
    train = pd.read_csv(dataset_dir / "local_train.csv")
    index = pd.read_csv(dataset_dir / "local_forecast_index_validation.csv")
    targets = pd.read_csv(dataset_dir / "local_validation_targets.csv")
    output_dir = ROOT / "output/local_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, predictions in make_all_baselines(train, index).items():
        prediction_path = output_dir / f"{name}.csv"
        predictions.to_csv(prediction_path, index=False)
        evaluate_file(targets, predictions, output_dir / f"{name}_metrics.json", name)
    print(f"Wrote and evaluated all baselines in {output_dir}")


if __name__ == "__main__":
    main()
