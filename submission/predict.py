"""Offline inference for multivariate rolling forecasts."""

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
    return ((values - mean) / scale).fillna(0.0).to_numpy(np.float32)


def forecast(
    history: pd.DataFrame,
    forecast_index: pd.DataFrame,
    checkpoint: dict,
    future_features: pd.DataFrame,
) -> pd.DataFrame:
    config = checkpoint["config"]
    preprocessing = checkpoint["preprocessing"]
    model = build_model(config)
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
    outputs = []

    with torch.no_grad():
        for series_id, requested in requested_features.groupby("series_id", sort=False):
            requested = requested.sort_values("timestamp")
            series_history = history.loc[
                history["series_id"].eq(series_id)
            ].sort_values("timestamp")
            if len(series_history) < context_length:
                raise ValueError(f"Series {series_id!r} has insufficient history")
            mean, scale = preprocessing["target_statistics"][str(series_id)]
            targets = list((series_history["target"].to_numpy(float) - mean) / scale)
            features = list(_transform_features(series_history, preprocessing))
            future = _transform_features(requested, preprocessing)
            predictions = []
            for offset in range(0, len(requested), prediction_length):
                block_features = future[offset : offset + prediction_length]
                actual_length = len(block_features)
                if actual_length < prediction_length:
                    block_features = np.concatenate(
                        [
                            block_features,
                            np.repeat(
                                block_features[-1:],
                                prediction_length - actual_length,
                                axis=0,
                            ),
                        ]
                    )
                x = torch.tensor(
                    targets[-context_length:], dtype=torch.float32
                ).unsqueeze(0)
                history_x = torch.tensor(
                    np.asarray(features[-context_length:]), dtype=torch.float32
                ).unsqueeze(0)
                future_x = torch.from_numpy(block_features).unsqueeze(0)
                model_inputs = {
                    "history_features": history_x,
                    "future_features": future_x,
                }
                if config.get("use_series_embedding"):
                    mapping = checkpoint["series_mapping"]
                    model_inputs["series_index"] = torch.tensor(
                        [mapping[str(series_id)]]
                    )
                block = model(x, **model_inputs).squeeze(0).numpy()
                targets.extend(block[:actual_length].tolist())
                features.extend(block_features[:actual_length].tolist())
                predictions.extend((block[:actual_length] * scale + mean).tolist())
            result = requested[keys].copy()
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
