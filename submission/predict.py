"""Standalone inference entry point for the final operations TCN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import (
    CHANNELS,
    CONTEXT_LENGTH,
    DROPOUT,
    EMBEDDING_SIZE,
    KERNEL_SIZE,
    LEVELS,
    PREDICTION_LENGTH,
    RESIDUAL_PERIOD,
    FinalTCN,
)


INFERENCE_BATCH_SIZE = 32
PREDICTION_FLOOR = 0.0
KEYS = ["series_id", "timestamp"]


def _select_input(input_dir: Path, test_name: str, validation_name: str) -> Path:
    test_path = input_dir / test_name
    if test_path.exists():
        return test_path
    validation_path = input_dir / validation_name
    if validation_path.exists():
        return validation_path
    raise FileNotFoundError(
        f"Expected {test_name!r} or {validation_name!r} in {input_dir}"
    )


def _transform_features(frame: pd.DataFrame, preprocessing: dict) -> np.ndarray:
    columns = preprocessing["feature_columns"]
    values = frame[columns].astype(float)
    mean = pd.Series(preprocessing["feature_mean"])[columns]
    scale = pd.Series(preprocessing["feature_scale"])[columns]
    standardized = ((values - mean) / scale).fillna(0.0).to_numpy(np.float32)
    missing_columns = preprocessing.get("missing_feature_columns", [])
    if not missing_columns:
        return standardized
    missingness = frame[missing_columns].isna().to_numpy(np.float32)
    return np.concatenate([standardized, missingness], axis=1)


def _validate_checkpoint(checkpoint: dict) -> None:
    config = checkpoint.get("config", {})
    expected = {
        "model": "tcn",
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "channels": CHANNELS,
        "levels": LEVELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "embedding_size": EMBEDDING_SIZE,
        "residual_period": RESIDUAL_PERIOD,
        "use_series_embedding": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {required!r})"
            for key, (actual, required) in mismatches.items()
        )
        raise ValueError(f"Checkpoint is not the selected final TCN: {details}")
    for key in ("preprocessing", "series_mapping", "state_dict"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing {key!r}")


def _forecast(
    history: pd.DataFrame,
    forecast_index: pd.DataFrame,
    future_features: pd.DataFrame,
    checkpoint: dict,
) -> pd.DataFrame:
    _validate_checkpoint(checkpoint)
    preprocessing = checkpoint["preprocessing"]
    series_mapping = checkpoint["series_mapping"]
    num_features = len(preprocessing["feature_columns"]) + len(
        preprocessing.get("missing_feature_columns", [])
    )
    model = FinalTCN(
        num_features=num_features,
        num_series=len(series_mapping),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    if forecast_index.duplicated(KEYS).any():
        raise ValueError("Forecast index contains duplicate series/timestamp rows")
    if future_features.duplicated(KEYS).any():
        raise ValueError("Future input contains duplicate series/timestamp rows")
    requested_features = forecast_index[KEYS].merge(
        future_features,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if requested_features["_merge"].ne("both").any():
        raise ValueError("Future input does not cover every forecast-index row")
    requested_features = requested_features.drop(columns="_merge")
    missing = set(preprocessing["feature_columns"]).difference(
        requested_features.columns
    )
    if missing:
        raise ValueError(f"Future input is missing features: {sorted(missing)}")

    counts = requested_features.groupby("series_id", sort=False).size()
    if not counts.eq(PREDICTION_LENGTH).all():
        raise ValueError(
            f"Final TCN requires exactly {PREDICTION_LENGTH} forecast rows per series"
        )

    prepared = []
    for series_id, requested in requested_features.groupby("series_id", sort=False):
        key = str(series_id)
        if key not in series_mapping:
            raise ValueError(f"Unknown series_id in forecast index: {series_id!r}")
        series_history = history.loc[history["series_id"].eq(series_id)].sort_values(
            "timestamp"
        )
        if len(series_history) < CONTEXT_LENGTH:
            raise ValueError(f"Series {series_id!r} has insufficient history")
        requested = requested.sort_values("timestamp")
        target_mean, target_scale = preprocessing["target_statistics"][key]
        normalized_target = (
            series_history["target"].to_numpy(float) - target_mean
        ) / target_scale
        prepared.append(
            {
                "requested": requested,
                "target_mean": float(target_mean),
                "target_scale": float(target_scale),
                "series_index": int(series_mapping[key]),
                "target": torch.tensor(
                    normalized_target[-CONTEXT_LENGTH:], dtype=torch.float32
                ),
                "history_features": torch.from_numpy(
                    _transform_features(series_history, preprocessing)[-CONTEXT_LENGTH:]
                ),
                "future_features": torch.from_numpy(
                    _transform_features(requested, preprocessing)
                ),
            }
        )

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(prepared), INFERENCE_BATCH_SIZE):
            batch = prepared[start : start + INFERENCE_BATCH_SIZE]
            normalized = (
                model(
                    torch.stack([item["target"] for item in batch]).to(device),
                    series_index=torch.tensor(
                        [item["series_index"] for item in batch], device=device
                    ),
                    history_features=torch.stack(
                        [item["history_features"] for item in batch]
                    ).to(device),
                    future_features=torch.stack(
                        [item["future_features"] for item in batch]
                    ).to(device),
                )
                .cpu()
                .numpy()
            )
            for item, normalized_series in zip(batch, normalized, strict=True):
                values = (
                    normalized_series * item["target_scale"] + item["target_mean"]
                )
                result = item["requested"][KEYS].copy()
                result["prediction"] = np.maximum(values, PREDICTION_FLOOR)
                outputs.append(result)

    unordered = pd.concat(outputs, ignore_index=True)
    predictions = forecast_index[KEYS].merge(
        unordered, on=KEYS, how="left", validate="one_to_one"
    )
    if len(predictions) != len(forecast_index) or not np.isfinite(
        predictions["prediction"]
    ).all():
        raise ValueError("Inference did not produce one finite prediction per row")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final TCN forecaster.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    forecast_index_path = _select_input(
        args.input_dir,
        "forecast_index_test.csv",
        "forecast_index_validation.csv",
    )
    future_input_path = _select_input(
        args.input_dir,
        "test_input.csv",
        "validation_input.csv",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predictions = _forecast(
        pd.read_csv(args.input_dir / "train.csv"),
        pd.read_csv(forecast_index_path),
        pd.read_csv(future_input_path),
        checkpoint,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
