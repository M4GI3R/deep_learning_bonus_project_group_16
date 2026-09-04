"""Create leakage-safe chronological local splits for a named dataset."""

from __future__ import annotations

import argparse

from .registry import (
    base_table_path,
    canonical_dataset_name,
    get_dataset_manifest,
    read_table,
    split_dir,
    split_table_path,
    write_table,
)


def create_split(dataset: str, horizon: int, *, force: bool = False) -> None:
    dataset = canonical_dataset_name(dataset)
    train_path = base_table_path(dataset, "train")
    if not train_path.exists():
        raise FileNotFoundError(
            f"Could not find {train_path}. Import it first with: "
            f"uv run python -m src.datasets.import_dataset {dataset}"
        )

    output_dir = split_dir(dataset, horizon)
    required = [
        split_table_path(dataset, horizon, "local_train"),
        split_table_path(dataset, horizon, "local_validation_input"),
        split_table_path(dataset, horizon, "local_validation_targets"),
        split_table_path(dataset, horizon, "local_forecast_index_validation"),
    ]
    if all(path.exists() for path in required) and not force:
        print(f"Local split already exists at {output_dir}")
        return

    print(f"Loading {dataset} from {train_path}")
    frame = read_table(train_path)
    required_columns = {"series_id", "timestamp", "target"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Training table is missing columns: {sorted(missing)}")
    frame = frame.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    counts = frame.groupby("series_id", sort=False).size()
    if (counts <= horizon).any():
        bad = counts[counts <= horizon]
        raise ValueError(
            f"Every series needs more than {horizon} rows; shortest have {bad.min()}"
        )

    validation = frame.groupby("series_id", sort=False).tail(horizon)
    train = frame.drop(validation.index).reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    write_table(train, required[0])
    write_table(validation.drop(columns="target"), required[1])
    write_table(validation[["series_id", "timestamp", "target"]], required[2])
    write_table(validation[["series_id", "timestamp"]], required[3])
    print(
        f"Prepared {len(train)} training and {len(validation)} validation rows "
        f"at {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="operations")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = get_dataset_manifest(args.dataset)
    horizon = args.horizon or int(manifest["validation_horizon"])
    create_split(args.dataset, horizon, force=args.force)


if __name__ == "__main__":
    main()
