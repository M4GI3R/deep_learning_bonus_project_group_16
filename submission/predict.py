"""Offline inference for direct multivariate forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import build_model


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


def forecast(
    history: pd.DataFrame,
    forecast_index: pd.DataFrame,
    checkpoint: dict,
    future_features: pd.DataFrame,
) -> pd.DataFrame:
    config = checkpoint["config"]
    preprocessing = checkpoint["preprocessing"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    keys = ["series_id", "timestamp"]
    if forecast_index.duplicated(keys).any() or future_features.duplicated(keys).any():
        raise ValueError(
            "Forecast index and future features require unique (series_id, timestamp) rows"
        )
    requested_features = forecast_index[keys].merge(
        future_features, on=keys, how="left", validate="one_to_one", indicator=True
    )
    if requested_features["_merge"].ne("both").any():
        raise ValueError(
            "Future input does not cover every requested (series_id, timestamp) row"
        )
    requested_features = requested_features.drop(columns="_merge")
    missing_columns = set(preprocessing["feature_columns"]).difference(
        requested_features.columns
    )
    if missing_columns:
        raise ValueError(
            f"Future input is missing feature columns: {sorted(missing_columns)}"
        )
    requested_counts = requested_features.groupby("series_id", sort=False).size()
    if (requested_counts > prediction_length).any():
        longest = int(requested_counts.max())
        raise ValueError(
            "Direct forecasting forbids autoregressive target rollout: "
            f"configured prediction_length={prediction_length}, requested={longest}. "
            "Train a model whose prediction length covers the complete index."
        )
    prepared = []
    mapping = checkpoint["series_mapping"]
    for series_id, requested in requested_features.groupby("series_id", sort=False):
        requested = requested.sort_values("timestamp")
        series_history = history.loc[history["series_id"].eq(series_id)].sort_values(
            "timestamp"
        )
        if len(series_history) < context_length:
            raise ValueError(f"Series {series_id!r} has insufficient history")
        mean, scale = preprocessing["target_statistics"][str(series_id)]
        targets = (series_history["target"].to_numpy(float) - mean) / scale
        features = _transform_features(series_history, preprocessing)
        future = _transform_features(requested, preprocessing)
        actual_length = len(future)
        if actual_length < prediction_length:
            future = np.concatenate(
                [
                    future,
                    np.repeat(
                        future[-1:], prediction_length - actual_length, axis=0
                    ),
                ]
            )
        prepared.append(
            {
                "requested": requested,
                "mean": float(mean),
                "scale": float(scale),
                "actual_length": actual_length,
                "series_index": mapping[str(series_id)],
                "target": torch.tensor(
                    targets[-context_length:], dtype=torch.float32
                ),
                "history_features": torch.tensor(
                    features[-context_length:], dtype=torch.float32
                ),
                "future_features": torch.from_numpy(future),
            }
        )

    outputs = []
    inference_batch_size = int(config.get("inference_batch_size", 32))
    with torch.inference_mode():
        for start in range(0, len(prepared), inference_batch_size):
            batch = prepared[start : start + inference_batch_size]
            model_inputs = {
                "history_features": torch.stack(
                    [item["history_features"] for item in batch]
                ).to(device),
                "future_features": torch.stack(
                    [item["future_features"] for item in batch]
                ).to(device),
            }
            if config.get("use_series_embedding"):
                model_inputs["series_index"] = torch.tensor(
                    [item["series_index"] for item in batch], device=device
                )
            normalized_batch = (
                model(
                    torch.stack([item["target"] for item in batch]).to(device),
                    **model_inputs,
                )
                .cpu()
                .numpy()
            )
            for item, normalized in zip(batch, normalized_batch, strict=True):
                actual_length = item["actual_length"]
                predictions = (
                    normalized[:actual_length] * item["scale"] + item["mean"]
                )
                prediction_floor = config.get("prediction_floor")
                if prediction_floor is not None:
                    predictions = np.maximum(predictions, float(prediction_floor))
                result = item["requested"][keys].copy()
                result["prediction"] = predictions
                outputs.append(result)

    result = pd.concat(outputs, ignore_index=True)
    if (
        len(result) != len(forecast_index)
        or not np.isfinite(result["prediction"]).all()
    ):
        raise ValueError(
            "Inference did not produce one finite prediction per requested row"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    forecast_index_name = (
        "forecast_index_test.csv"
        if (args.input_dir / "forecast_index_test.csv").exists()
        else "forecast_index_validation.csv"
    )
    future_name = (
        "test_input.csv"
        if (args.input_dir / "test_input.csv").exists()
        else "validation_input.csv"
    )
    history = pd.read_csv(args.input_dir / "train.csv")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    predictions = forecast(
        history,
        pd.read_csv(args.input_dir / forecast_index_name),
        checkpoint,
        pd.read_csv(args.input_dir / future_name),
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
