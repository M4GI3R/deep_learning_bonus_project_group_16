"""Small YAML configuration loader with Hydra-style key=value overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    config = deepcopy(config)
    # Defaults keep configs and checkpoints created before multi-dataset support
    # compatible with the original operations dataset.
    config.setdefault("dataset", "operations_forecasting_2026")
    config.setdefault("feature_set", "provided")
    config.setdefault("training_objective", "overall")
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown configuration key {dotted_key!r}")
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"Unknown configuration key {dotted_key!r}")
        target[keys[-1]] = yaml.safe_load(raw_value)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "dataset",
        "feature_set",
        "training_objective",
        "run_name",
        "seed",
        "context_length",
        "prediction_length",
        "stride",
        "max_epochs",
        "batch_size",
        "inference_batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "selection_metric",
        "prediction_floor",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "model",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if not str(config["dataset"]).strip():
        raise ValueError("dataset must be non-empty")
    if not str(config["feature_set"]).strip():
        raise ValueError("feature_set must be non-empty")
    if config["training_objective"] not in {"overall", "wape"}:
        raise ValueError("training_objective must be either 'overall' or 'wape'")
    model_type = config["model"].get("type")
    if model_type not in {"dlinear", "tcn"}:
        raise ValueError("model.type must be either 'dlinear' or 'tcn'")
    if config["selection_metric"] not in {
        "Overall",
        "MAE",
        "MSE",
        "RMSE",
        "MAPE (%)",
        "sMAPE (%)",
        "WAPE",
    }:
        raise ValueError("selection_metric must be a leaderboard metric")
    if (
        not str(config["run_name"]).strip()
        or Path(str(config["run_name"])).name != config["run_name"]
    ):
        raise ValueError("run_name must be a non-empty directory name")
    for key in (
        "context_length",
        "prediction_length",
        "stride",
        "max_epochs",
        "batch_size",
        "inference_batch_size",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    if float(config["gradient_clip_norm"]) <= 0:
        raise ValueError("gradient_clip_norm must be positive")


def write_config(config: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False))
