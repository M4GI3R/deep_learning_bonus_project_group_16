import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.registry import (
    canonical_dataset_name,
    get_dataset_manifest,
    read_table,
    split_table_path,
)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    # MAPE is undefined where the actual value is zero. Electricity contains
    # legitimate zero loads, so evaluate MAPE on non-zero actuals rather than
    # dividing by an arbitrary epsilon and producing meaningless huge values.
    nonzero_actual = np.abs(y_true) > 1e-8
    if nonzero_actual.any():
        mape = float(
            np.mean(
                np.abs(
                    (y_true[nonzero_actual] - y_pred[nonzero_actual])
                    / np.abs(y_true[nonzero_actual])
                )
            )
            * 100
        )
    else:
        mape = 0.0 if np.all(np.abs(y_pred) <= 1e-8) else float("inf")
    smape = float(
        200
        * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    )
    wape = float(
        np.sum(np.abs(y_true - y_pred))
        / max(float(np.sum(np.abs(y_true))), 1e-8)
        * 100.0
    )
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "sMAPE (%)": smape,
        "WAPE": wape,
    }


def calculate_overall_proxy(y_true, y_pred):
    """Return a scale-free local proxy for the public percentile-rank aggregate.

    The public leaderboard percentile-ranks all six metrics across submissions.
    A single checkpoint cannot know those percentiles, so local model selection
    averages scale-free versions of the same six metrics instead.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = calculate_metrics(y_true, y_pred)
    mean_abs_target = float(np.mean(np.abs(y_true))) + 1e-8
    mean_square_target = float(np.mean(y_true**2)) + 1e-8
    rms_target = float(np.sqrt(mean_square_target)) + 1e-8
    components = [
        metrics["MAE"] / mean_abs_target,
        metrics["MSE"] / mean_square_target,
        metrics["RMSE"] / rms_target,
        metrics["MAPE (%)"] / 100.0,
        metrics["sMAPE (%)"] / 100.0,
        metrics["WAPE"] / 100.0,
    ]
    return float(np.mean(components))


def validate_prediction_contract(gt_df, pred_df, pred_col, display_name):
    """Reject incomplete or ambiguous forecasts before calculating metrics."""
    keys = ["series_id", "timestamp"]
    if gt_df.duplicated(keys).any():
        raise ValueError("ground truth contains duplicate (series_id, timestamp) rows")
    if pred_df.duplicated(keys).any():
        raise ValueError("predictions contain duplicate (series_id, timestamp) rows")
    if not pd.api.types.is_numeric_dtype(pred_df[pred_col]):
        raise ValueError(f"{pred_col!r} must be numeric")
    if not np.isfinite(pred_df[pred_col].to_numpy()).all():
        raise ValueError(f"{pred_col!r} contains missing or non-finite values")

    expected = pd.MultiIndex.from_frame(gt_df[keys])
    actual = pd.MultiIndex.from_frame(pred_df[keys])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing) or len(extra) or len(pred_df) != len(gt_df):
        raise ValueError(
            f"prediction coverage mismatch for {display_name}: "
            f"expected={len(gt_df)}, actual={len(pred_df)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )


def evaluate_file(
    gt_df,
    pred_df,
    metrics_path,
    display_name,
    *,
    best_epoch=None,
    category="Root",
    name=None,
):
    """Evaluate one prediction frame and write dashboard-compatible metrics."""
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    gt_df["timestamp"] = pd.to_datetime(gt_df["timestamp"])
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    validate_prediction_contract(gt_df, pred_df, "prediction", display_name)
    aligned = gt_df.merge(pred_df, on=["series_id", "timestamp"], validate="one_to_one")
    aligned = aligned.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    global_metrics = calculate_metrics(
        aligned["target"].to_numpy(), aligned["prediction"].to_numpy()
    )

    per_series_smape = {}
    for series_id, group in aligned.groupby("series_id", sort=False):
        y_true = group["target"].to_numpy()
        y_pred = group["prediction"].to_numpy()
        per_series_smape[str(series_id)] = float(
            200
            * np.mean(
                np.abs(y_true - y_pred)
                / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
            )
        )

    aligned["step"] = aligned.groupby("series_id").cumcount() + 1
    horizon_mae = {
        str(int(step)): float(
            np.mean(np.abs(group["target"] - group["prediction"]))
        )
        for step, group in aligned.groupby("step", sort=True)
    }

    payload = {
        "metric_scale": "leaderboard_percent_v1",
        "mape_zero_policy": "exclude_zero_actuals",
        "name": name or display_name,
        "category": category,
        "display_name": display_name,
        "global_metrics": global_metrics,
        "per_series_smape": per_series_smape,
        "horizon_mae": horizon_mae,
    }
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="operations")
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset = canonical_dataset_name(args.dataset)
    manifest = get_dataset_manifest(dataset)
    horizon = args.horizon or int(manifest["validation_horizon"])
    output_dir = project_root / "output" / dataset
    gt_path = split_table_path(dataset, horizon, "local_validation_targets")

    if not gt_path.exists():
        print(
            f"Error: Local validation targets ground truth file not found at {gt_path}"
        )
        print(
            "Please split the dataset first: uv run python -m src.datasets.split_data "
            f"--dataset {dataset} --horizon {horizon}"
        )
        return

    print(f"Loading local ground truth targets from {gt_path}...")
    gt_df = read_table(gt_path)
    gt_df["timestamp"] = pd.to_datetime(gt_df["timestamp"])

    # Discover local-holdout prediction CSV files for the selected dataset.
    # Full-training directories contain public-validation exports whose hidden
    # targets are unavailable locally and must never be compared to this holdout.
    print(f"Scanning {output_dir} for prediction files...")
    prediction_files = []

    for file in output_dir.rglob("*.csv"):
        relative_file = file.relative_to(output_dir)
        if any(part.endswith("_full_training") for part in relative_file.parts[:-1]):
            continue
        # Ignore files that are obviously not predictions
        if file.name.startswith("metrics"):
            continue

        try:
            # Quick validation of schema
            df_head = pd.read_csv(file, nrows=2)
            required_cols = {"series_id", "timestamp"}
            if not required_cols.issubset(df_head.columns):
                continue
            # Must have either "prediction" or "target" (ground truth)
            if "prediction" not in df_head.columns and "target" not in df_head.columns:
                continue

            # Determine directory structure
            if file.name.lower() in ["predictions.csv", "prediction.csv"]:
                # Nested layout: output/<category>/<model_name>/predictions.csv
                model_name = file.parent.name
                parent = file.parent.parent.relative_to(output_dir)
                category = (
                    dataset
                    if str(parent) == "."
                    else f"{dataset} / {parent.as_posix().replace('/', ' / ')}"
                )
                metrics_path = file.parent / "metrics.json"
            else:
                # Flat layout: output/<category>/<model_name>.csv
                model_name = file.stem
                parent = file.parent.relative_to(output_dir)
                category = (
                    dataset
                    if str(parent) == "."
                    else f"{dataset} / {parent.as_posix().replace('/', ' / ')}"
                )
                metrics_path = file.parent / f"{file.stem}_metrics.json"

            display_name = f"{category} / {model_name}"
            existing_best_epoch = None
            if metrics_path.exists():
                try:
                    existing_best_epoch = json.loads(
                        metrics_path.read_text()
                    ).get("best_epoch")
                except (OSError, ValueError):
                    pass

            prediction_files.append(
                {
                    "name": model_name,
                    "category": category,
                    "display_name": display_name,
                    "path": file,
                    "metrics_path": metrics_path,
                    "best_epoch": existing_best_epoch,
                }
            )
        except Exception:
            # Not a valid CSV or permission error
            continue

    if not prediction_files:
        print("No prediction files found in the output directory.")
        return

    print(f"Found {len(prediction_files)} prediction files. Evaluating...")

    evaluated = 0
    for pred in prediction_files:
        print(f"Evaluating {pred['display_name']}...")
        try:
            pred_df = pd.read_csv(pred["path"])
            pred_col = "prediction" if "prediction" in pred_df.columns else "target"
            canonical_predictions = pred_df[
                ["series_id", "timestamp", pred_col]
            ].rename(columns={pred_col: "prediction"})
            evaluate_file(
                gt_df,
                canonical_predictions,
                pred["metrics_path"],
                pred["display_name"],
                best_epoch=pred["best_epoch"],
                category=pred["category"],
                name=pred["name"],
            )
            evaluated += 1
            print(f"Saved metrics to {pred['metrics_path']}")
        except Exception as e:
            print(f"Error evaluating {pred['display_name']}: {e}")

    print(f"Metric precomputation complete: {evaluated} files evaluated.")


if __name__ == "__main__":
    main()
