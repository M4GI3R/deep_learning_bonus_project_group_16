"""Dataset schema, preprocessing, and sliding training windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datasets.registry import feature_columns as dataset_feature_columns

# Backward-compatible default for tests and older callers. New training runs pass
# the selected dataset manifest's columns explicitly.
FEATURE_COLUMNS = dataset_feature_columns("operations_forecasting_2026", "provided")


def validate_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool,
    feature_columns: list[str] | None = None,
) -> None:
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    required = {"series_id", "timestamp", *columns}
    if require_target:
        required.add("target")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    core = ["series_id", "timestamp"] + (["target"] if require_target else [])
    if frame[core].isna().any().any():
        raise ValueError(f"Missing values are not allowed in {core}")


def fit_preprocessing(
    frame: pd.DataFrame, feature_columns: list[str] | None = None
) -> dict:
    columns = FEATURE_COLUMNS if feature_columns is None else list(feature_columns)
    validate_frame(frame, require_target=True, feature_columns=columns)
    feature_mean = frame[columns].mean().fillna(0.0)
    feature_scale = frame[columns].std().fillna(1.0).clip(lower=1e-6)
    missing_feature_columns = [
        column for column in columns if frame[column].isna().any()
    ]
    target_statistics = {}
    for series_id, part in frame.groupby("series_id", sort=False):
        values = part["target"].to_numpy(float)
        scale = float(values.std())
        target_statistics[str(series_id)] = (
            float(values.mean()),
            scale if scale > 1e-6 else 1.0,
        )
    return {
        "feature_columns": columns,
        "feature_mean": feature_mean.to_dict(),
        "feature_scale": feature_scale.to_dict(),
        "missing_feature_columns": missing_feature_columns,
        "target_statistics": target_statistics,
    }


def transform_features(frame: pd.DataFrame, preprocessing: dict) -> np.ndarray:
    columns = preprocessing["feature_columns"]
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float32)
    values = frame[columns].astype(float)
    mean = pd.Series(preprocessing["feature_mean"])[columns]
    scale = pd.Series(preprocessing["feature_scale"])[columns]
    standardized = ((values - mean) / scale).fillna(0.0).to_numpy(np.float32)
    missing_columns = preprocessing.get("missing_feature_columns", [])
    if not missing_columns:
        return standardized
    # Missing forecast covariates are not missing completely at random.  Preserve
    # the safe mean imputation while exposing explicit indicators to the model.
    missingness = frame[missing_columns].isna().to_numpy(np.float32)
    return np.concatenate([standardized, missingness], axis=1)


def transformed_feature_count(preprocessing: dict) -> int:
    """Return the number of model inputs after adding missingness indicators."""
    return len(preprocessing["feature_columns"]) + len(
        preprocessing.get("missing_feature_columns", [])
    )


class WindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        context: int,
        horizon: int,
        stride: int,
        preprocessing: dict,
        series_mapping: dict[str, int],
    ) -> None:
        validate_frame(
            frame,
            require_target=True,
            feature_columns=preprocessing["feature_columns"],
        )
        self.context, self.horizon = context, horizon
        self.series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.target_means: dict[str, float] = {}
        self.target_scales: dict[str, float] = {}
        self.series_mapping = series_mapping
        self.windows: list[tuple[str, int]] = []
        for series_id, part in frame.groupby("series_id", sort=False):
            part = part.sort_values("timestamp")
            key = str(series_id)
            values = part["target"].to_numpy(np.float32)
            mean, scale = preprocessing["target_statistics"][key]
            self.target_means[key] = float(mean)
            self.target_scales[key] = float(scale)
            self.series[key] = (
                (values - mean) / scale,
                transform_features(part, preprocessing),
            )
            for start in range(0, len(values) - context - horizon + 1, stride):
                self.windows.append((key, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        series_id, start = self.windows[index]
        targets, features = self.series[series_id]
        split = start + self.context
        return (
            torch.from_numpy(targets[start:split]),
            torch.from_numpy(targets[split : split + self.horizon]),
            torch.tensor(self.series_mapping[series_id]),
            torch.from_numpy(features[start:split]),
            torch.from_numpy(features[split : split + self.horizon]),
            torch.tensor(self.target_means[series_id], dtype=torch.float32),
            torch.tensor(self.target_scales[series_id], dtype=torch.float32),
        )
