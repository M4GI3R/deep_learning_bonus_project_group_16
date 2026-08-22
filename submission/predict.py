"""Offline inference entrypoint for validation and private evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import build_model


def load_history(input_dir: Path) -> pd.DataFrame:
    for name in ("test_input.csv", "validation_input.csv", "train.csv"):
        path = input_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            if "target" in frame.columns:
                return frame
    raise FileNotFoundError("No test_input.csv, validation_input.csv, or train.csv with a target column found")


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    for name in ("forecast_index_test.csv", "forecast_index_validation.csv"):
        path = input_dir / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No forecast_index_test.csv or forecast_index_validation.csv found")


def forecast(history: pd.DataFrame, forecast_index: pd.DataFrame, checkpoint: dict) -> pd.DataFrame:
    config = checkpoint["config"]
    model = build_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    context_length = config["context_length"]
    statistics = checkpoint["series_statistics"]
    outputs = []

    with torch.no_grad():
        for series_id, requested in forecast_index.groupby("series_id", sort=False):
            requested = requested.sort_values("timestamp")
            values = history.loc[history["series_id"].eq(series_id)].sort_values("timestamp")["target"].to_numpy(float)
            if len(values) < context_length:
                raise ValueError(f"Series {series_id!r} has {len(values)} history rows; {context_length} required")
            mean, scale = statistics[str(series_id)]
            normalized = list((values - mean) / scale)
            predictions = []
            while len(predictions) < len(requested):
                x = torch.tensor(normalized[-context_length:], dtype=torch.float32).unsqueeze(0)
                block = model(x).squeeze(0).numpy()
                normalized.extend(block.tolist())
                predictions.extend((block * scale + mean).tolist())
            result = requested[["series_id", "timestamp"]].copy()
            result["prediction"] = predictions[:len(result)]
            outputs.append(result)

    result = pd.concat(outputs, ignore_index=True)
    if len(result) != len(forecast_index) or not np.isfinite(result["prediction"]).all():
        raise ValueError("Inference did not produce one finite prediction per requested row")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predictions = forecast(load_history(args.input_dir), load_forecast_index(args.input_dir), checkpoint)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
