"""Reproducibly download and prepare either supported forecasting dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .registry import (
    base_table_path,
    canonical_dataset_name,
    dataset_dir,
    get_dataset_manifest,
    processed_dir,
)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def import_operations(force: bool) -> None:
    manifest = get_dataset_manifest("operations_forecasting_2026")
    root = dataset_dir(manifest["name"])
    raw_dir = root / "raw"
    target_dir = processed_dir(manifest["name"])
    expected = [
        target_dir / "train.csv",
        target_dir / "validation_input.csv",
        target_dir / "forecast_index_validation.csv",
        target_dir / "metadata.json",
    ]
    if all(path.exists() for path in expected) and not force:
        print(f"Operations dataset is already available at {target_dir}")
        return

    from huggingface_hub import snapshot_download

    print(f"Downloading {manifest['source']['repo_id']} to {raw_dir}")
    snapshot_download(
        repo_id=manifest["source"]["repo_id"],
        repo_type="dataset",
        revision=manifest["source"]["revision"],
        local_dir=raw_dir,
        ignore_patterns=[".git*"],
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in raw_dir.iterdir():
        if source.is_file() and source.suffix in {".csv", ".json"}:
            destination = target_dir / source.name
            if force and destination.exists():
                destination.unlink()
            _link_or_copy(source, destination)
    print(f"Prepared operations dataset at {target_dir}")


def _write_electricity_parquet(wide: pd.DataFrame, destination: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    writer = None
    try:
        timestamps = wide.index.to_numpy()
        for series_id in wide.columns:
            frame = pd.DataFrame(
                {
                    "series_id": str(series_id),
                    "timestamp": timestamps,
                    "target": wide[series_id].to_numpy(dtype="float32"),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(destination)


def import_electricity(force: bool) -> None:
    manifest = get_dataset_manifest("electricity_load_diagrams")
    root = dataset_dir(manifest["name"])
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "electricityloaddiagrams20112014.zip"
    target_path = base_table_path(manifest["name"], "train")
    if target_path.exists() and not force:
        print(f"Electricity dataset is already available at {target_path}")
        return
    if not archive_path.exists():
        print(f"Downloading the original UCI archive to {archive_path}")
        partial = archive_path.with_suffix(archive_path.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        request = urllib.request.Request(
            manifest["source"]["url"], headers={"User-Agent": "group-16-forecasting/1.0"}
        )
        with urllib.request.urlopen(request) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(archive_path)

    print("Reproducing the Hugging Face 'lstnet' hourly target representation")
    with zipfile.ZipFile(archive_path) as archive:
        member = next(
            name
            for name in archive.namelist()
            if name.endswith("LD2011_2014.txt") and not name.startswith("__MACOSX/")
        )
        with archive.open(member) as stream:
            wide = pd.read_csv(
                stream,
                sep=";",
                index_col=0,
                parse_dates=True,
                decimal=",",
            )
    wide.sort_index(inplace=True)
    wide = wide.resample("1h").sum()
    wide = wide[(wide.index.year >= 2012) & (wide.index.year <= 2014)]
    # This is the exact client filtering used by the Hugging Face lstnet config.
    wide = wide.T[wide.iloc[0] > 0].T.astype("float32")
    if len(wide) != int(manifest["trend_periods"]):
        raise ValueError(
            f"Expected {manifest['trend_periods']} hourly steps, got {len(wide)}"
        )
    if wide.shape[1] != int(manifest["expected_series"]):
        raise ValueError(
            f"Expected {manifest['expected_series']} active clients, got {wide.shape[1]}"
        )

    _write_electricity_parquet(wide, target_path)
    metadata = {
        "name": manifest["name"],
        "source": manifest["source"],
        "frequency": "h",
        "n_series": int(wide.shape[1]),
        "n_steps": int(wide.shape[0]),
        "target": "Hugging Face hourly-resampled load value (unchanged)",
        "start": str(wide.index.min()),
        "end": str(wide.index.max()),
    }
    (processed_dir(manifest["name"]) / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Prepared {wide.shape[1]} hourly series at {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="operations or electricity")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset = canonical_dataset_name(args.dataset)
    if dataset == "operations_forecasting_2026":
        import_operations(args.force)
    elif dataset == "electricity_load_diagrams":
        import_electricity(args.force)
    else:
        raise ValueError(f"No importer implemented for {dataset}")


if __name__ == "__main__":
    main()
