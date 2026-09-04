"""One-command local baseline generation and evaluation for a named dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "res/provided_res/baseline"))

from baselines import make_all_baselines  # noqa: E402
from src.datasets.registry import (  # noqa: E402
    canonical_dataset_name,
    get_dataset_manifest,
    read_table,
    split_table_path,
)
from src.evaluate_predictions import evaluate_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="operations")
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    dataset = canonical_dataset_name(args.dataset)
    manifest = get_dataset_manifest(dataset)
    horizon = args.horizon or int(manifest["validation_horizon"])
    required = [
        split_table_path(dataset, horizon, "local_train"),
        split_table_path(dataset, horizon, "local_validation_targets"),
        split_table_path(dataset, horizon, "local_forecast_index_validation"),
    ]
    if not all(path.exists() for path in required):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.datasets.split_data",
                "--dataset",
                dataset,
                "--horizon",
                str(horizon),
            ],
            check=True,
        )
    train = read_table(required[0])
    targets = read_table(required[1])
    index = read_table(required[2])
    output_dir = ROOT / "output" / dataset / "baselines" / f"horizon_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, predictions in make_all_baselines(train, index).items():
        prediction_path = output_dir / f"{name}.csv"
        predictions.to_csv(prediction_path, index=False)
        evaluate_file(
            targets,
            predictions,
            output_dir / f"{name}_metrics.json",
            f"{dataset} / baselines / horizon_{horizon} / {name}",
            category=f"{dataset} / baselines / horizon_{horizon}",
            name=name,
        )
    print(f"Wrote and evaluated all baselines in {output_dir}")


if __name__ == "__main__":
    main()
