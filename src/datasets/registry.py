"""Named dataset manifests, paths, table I/O, and feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = ROOT / "configs" / "datasets"
DATASET_ROOT = ROOT / "res" / "datasets"
OUTPUT_ROOT = ROOT / "output"


def _manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in sorted(DATASET_CONFIG_DIR.glob("*.yaml")):
        manifest = yaml.safe_load(path.read_text())
        if not isinstance(manifest, dict) or "name" not in manifest:
            raise ValueError(f"Invalid dataset manifest: {path}")
        manifest["_path"] = path
        manifests[str(manifest["name"])] = manifest
    return manifests


def get_dataset_manifest(name: str) -> dict[str, Any]:
    requested = str(name).strip().lower()
    for canonical, manifest in _manifests().items():
        aliases = {canonical.lower()}
        aliases.update(str(value).lower() for value in manifest.get("aliases", []))
        if requested in aliases:
            return manifest
    available = ", ".join(sorted(_manifests()))
    raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}")


def canonical_dataset_name(name: str) -> str:
    return str(get_dataset_manifest(name)["name"])


def resolve_feature_set(dataset: str, feature_set: str | None) -> str:
    manifest = get_dataset_manifest(dataset)
    selected = feature_set or manifest["default_feature_set"]
    if selected == "default":
        selected = manifest["default_feature_set"]
    if selected not in manifest["feature_sets"]:
        available = ", ".join(manifest["feature_sets"])
        raise ValueError(
            f"Unknown feature set {selected!r} for {manifest['name']}. "
            f"Available feature sets: {available}"
        )
    return str(selected)


def feature_columns(dataset: str, feature_set: str | None) -> list[str]:
    manifest = get_dataset_manifest(dataset)
    selected = resolve_feature_set(dataset, feature_set)
    return list(manifest["feature_sets"][selected])


def dataset_dir(dataset: str) -> Path:
    return DATASET_ROOT / canonical_dataset_name(dataset)


def processed_dir(dataset: str) -> Path:
    return dataset_dir(dataset) / "processed"


def split_dir(dataset: str, horizon: int) -> Path:
    return processed_dir(dataset) / "splits" / f"horizon_{int(horizon)}"


def table_extension(dataset: str) -> str:
    return str(get_dataset_manifest(dataset).get("table_format", "csv"))


def base_table_path(dataset: str, stem: str) -> Path:
    return processed_dir(dataset) / f"{stem}.{table_extension(dataset)}"


def split_table_path(dataset: str, horizon: int, stem: str) -> Path:
    return split_dir(dataset, horizon) / f"{stem}.{table_extension(dataset)}"


def model_run_dir(dataset: str, feature_set: str | None, run_name: str) -> Path:
    """Return the canonical output directory for a neural-model run.

    The operations dataset has one supplied feature set, so an extra ``provided``
    directory carries no experimental information. Electricity retains its
    feature-set directory because it identifies the three ablation experiments.
    """
    dataset_name = canonical_dataset_name(dataset)
    selected = resolve_feature_set(dataset_name, feature_set)
    root = OUTPUT_ROOT / dataset_name
    if selected == "provided":
        return root / run_name
    return root / selected / run_name


def model_category(dataset: str, feature_set: str | None) -> str:
    """Return the dashboard category matching :func:`model_run_dir`."""
    dataset_name = canonical_dataset_name(dataset)
    selected = resolve_feature_set(dataset_name, feature_set)
    if selected == "provided":
        return dataset_name
    return f"{dataset_name} / {selected}"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path}")


def _engineer_electricity_features(
    frame: pd.DataFrame, requested_columns: list[str]
) -> pd.DataFrame:
    result = frame.copy(deep=False)
    timestamps = pd.to_datetime(result["timestamp"])
    hour_angle = 2.0 * np.pi * timestamps.dt.hour / 24.0
    dow_angle = 2.0 * np.pi * timestamps.dt.dayofweek / 7.0
    result["hour_sin"] = np.sin(hour_angle).astype("float32")
    result["hour_cos"] = np.cos(hour_angle).astype("float32")
    result["dow_sin"] = np.sin(dow_angle).astype("float32")
    result["dow_cos"] = np.cos(dow_angle).astype("float32")
    result["is_weekend"] = timestamps.dt.dayofweek.ge(5).astype("float32")

    # This matches the supplied operations feature concept: a deterministic,
    # globally standardized linear time index. It depends only on timestamps.
    manifest = get_dataset_manifest("electricity_load_diagrams")
    origin = pd.Timestamp(manifest["trend_origin"])
    periods = int(manifest["trend_periods"])
    positions = (timestamps - origin).dt.total_seconds() / 3600.0
    mean = (periods - 1) / 2.0
    scale = np.sqrt((periods**2 - 1) / 12.0)
    result["trend"] = ((positions - mean) / scale).astype("float32")

    years = sorted(int(year) for year in timestamps.dt.year.unique())
    if "day_of_year_sin" in requested_columns:
        days_in_year = np.where(timestamps.dt.is_leap_year, 366.0, 365.0)
        day_angle = 2.0 * np.pi * (timestamps.dt.dayofyear - 1) / days_in_year
        result["day_of_year_sin"] = np.sin(day_angle).astype("float32")
        result["day_of_year_cos"] = np.cos(day_angle).astype("float32")
    if "month_sin" in requested_columns:
        month_angle = 2.0 * np.pi * (timestamps.dt.month - 1) / 12.0
        result["month_sin"] = np.sin(month_angle).astype("float32")
        result["month_cos"] = np.cos(month_angle).astype("float32")
    if "is_portuguese_holiday" in requested_columns:
        try:
            import holidays
        except ImportError as exc:
            raise ImportError(
                "The extended electricity feature set requires the 'holidays' package. "
                "Run 'uv sync' before training."
            ) from exc
        portuguese_holidays = holidays.country_holidays("PT", years=years)
        holiday_dates = pd.to_datetime(list(portuguese_holidays)).normalize()
        result["is_portuguese_holiday"] = (
            timestamps.dt.normalize().isin(holiday_dates).astype("float32")
        )
    if "is_dst_transition" in requested_columns:
        is_last_sunday = timestamps.dt.dayofweek.eq(6) & (
            timestamps.dt.day + 7 > timestamps.dt.days_in_month
        )
        result["is_dst_transition"] = (
            timestamps.dt.month.isin([3, 10])
            & is_last_sunday
            & timestamps.dt.hour.eq(1)
        ).astype("float32")
    return result


def prepare_features(
    frame: pd.DataFrame, dataset: str, feature_set: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame with exactly the requested leakage-safe feature columns."""
    selected = resolve_feature_set(dataset, feature_set)
    columns = feature_columns(dataset, selected)
    result = frame.copy(deep=False)
    if canonical_dataset_name(dataset) == "electricity_load_diagrams" and columns:
        result = _engineer_electricity_features(result, columns)
    missing = set(columns).difference(result.columns)
    if missing:
        raise ValueError(
            f"Dataset {dataset!r}, feature set {selected!r} is missing columns: "
            f"{sorted(missing)}"
        )
    return result, columns
