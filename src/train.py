"""YAML-driven local training, early stopping, prediction, and evaluation."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))

from predict import forecast  # noqa: E402
from config import load_config, write_config  # noqa: E402
from data import WindowDataset, fit_preprocessing, transformed_feature_count  # noqa: E402
from evaluate_predictions import (  # noqa: E402
    calculate_metrics,
    calculate_overall_proxy,
    evaluate_file,
)
from src.model import build_model  # noqa: E402


def make_checkpoint(
    model: torch.nn.Module,
    config: dict,
    preprocessing: dict,
    series_mapping: dict[str, int],
    best_epoch: int,
) -> dict:
    return {
        "config": config,
        "preprocessing": preprocessing,
        "series_mapping": series_mapping,
        "best_epoch": best_epoch,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def train(
    config: dict,
    train_frame: pd.DataFrame,
    *,
    validation: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    output_dir: Path,
    epochs: int | None = None,
) -> Path:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    preprocessing = fit_preprocessing(train_frame)
    series_mapping = {
        str(value): index
        for index, value in enumerate(train_frame["series_id"].unique())
    }
    runtime_config = dict(config)
    runtime_config.update(
        {
            "num_features": transformed_feature_count(preprocessing),
            "num_series": len(series_mapping),
            "use_series_embedding": bool(
                config["model"].get("use_series_embedding", True)
            ),
        }
    )
    runtime_config.update(config["model"])
    runtime_config["model"] = config["model"]["type"]
    dataset = WindowDataset(
        train_frame,
        config["context_length"],
        config["prediction_length"],
        config["stride"],
        preprocessing,
        series_mapping,
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(runtime_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    max_epochs = epochs or config["max_epochs"]
    selection_metric = config["selection_metric"]
    best_score = float("inf")
    best_epoch = max_epochs
    stale_epochs = 0
    output_dir.mkdir(parents=True, exist_ok=False)
    write_config(config, output_dir / "config.yaml")
    checkpoint_path = output_dir / "checkpoint.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        for (
            x,
            y,
            series_index,
            history_features,
            future_features,
            target_mean,
            target_scale,
        ) in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x,
                series_index=series_index.to(device),
                history_features=history_features.to(device),
                future_features=future_features.to(device),
            )
            # Match the public "Overall" leaderboard: it percentile-ranks MAE,
            # MSE, RMSE, MAPE, sMAPE, and WAPE equally.  Percentile ranks are not
            # differentiable within one run, so optimize scale-free versions of
            # the same six quantities in the original target scale.
            scale = target_scale.to(device).unsqueeze(1)
            mean = target_mean.to(device).unsqueeze(1)
            raw_prediction = prediction * scale + mean
            raw_target = y * scale + mean
            absolute_error = torch.abs(raw_prediction - raw_target)
            squared_error = (raw_prediction - raw_target).square()
            epsilon = 1e-6
            mean_abs_target = raw_target.abs().mean().clamp_min(epsilon)
            mean_square_target = raw_target.square().mean().clamp_min(epsilon)
            rms_target = mean_square_target.sqrt().clamp_min(epsilon)
            loss_components = torch.stack(
                [
                    absolute_error.mean() / mean_abs_target,
                    squared_error.mean() / mean_square_target,
                    squared_error.mean().clamp_min(epsilon).sqrt() / rms_target,
                    (absolute_error / raw_target.abs().clamp_min(epsilon)).mean(),
                    (
                        2.0
                        * absolute_error
                        / (raw_target.abs() + raw_prediction.abs()).clamp_min(epsilon)
                    ).mean(),
                    absolute_error.sum() / raw_target.abs().sum().clamp_min(epsilon),
                ]
            )
            loss = loss_components.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"]
            )
            optimizer.step()
            total_loss += loss.item() * len(x)
            total_mae += absolute_error.mean().item() * len(x)
        print(
            f"epoch={epoch:03d} train_overall_proxy="
            f"{total_loss / len(dataset):.6f} "
            f"train_mae={total_mae / len(dataset):.6f}"
        )

        if validation is None:
            continue
        history, future, targets = validation
        checkpoint = make_checkpoint(
            model, runtime_config, preprocessing, series_mapping, epoch
        )
        predictions = forecast(
            history, targets[["series_id", "timestamp"]], checkpoint, future
        )
        joined = targets.merge(
            predictions, on=["series_id", "timestamp"], validate="one_to_one"
        )
        scores = calculate_metrics(
            joined["target"].to_numpy(), joined["prediction"].to_numpy()
        )
        overall_proxy = calculate_overall_proxy(
            joined["target"].to_numpy(), joined["prediction"].to_numpy()
        )
        score = (
            overall_proxy
            if selection_metric == "Overall"
            else scores[selection_metric]
        )
        print(
            f"validation_mae={scores['MAE']:.6f} "
            f"validation_mse={scores['MSE']:.6f} "
            f"validation_rmse={scores['RMSE']:.6f} "
            f"validation_mape_pct={scores['MAPE (%)']:.6f} "
            f"validation_smape={scores['sMAPE (%)']:.6f} "
            f"validation_wape_pct={scores['WAPE']:.6f} "
            f"validation_overall_proxy={overall_proxy:.6f}"
        )
        if score < best_score - config["early_stopping_min_delta"]:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save(checkpoint, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config["early_stopping_patience"]:
                print(
                    f"Early stopping at epoch {epoch}; best {selection_metric} "
                    f"was {best_score:.6f} at epoch {best_epoch}"
                )
                break

    if validation is None:
        torch.save(
            make_checkpoint(
                model, runtime_config, preprocessing, series_mapping, max_epochs
            ),
            checkpoint_path,
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        history, future, targets = validation
        predictions = forecast(
            history, targets[["series_id", "timestamp"]], checkpoint, future
        )
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        evaluate_file(
            targets,
            predictions,
            output_dir / "metrics.json",
            config["run_name"],
            best_epoch=best_epoch,
        )
    return checkpoint_path


def ensure_local_split() -> None:
    required = [
        ROOT / "res/dataset/local_train.csv",
        ROOT / "res/dataset/local_validation_input.csv",
        ROOT / "res/dataset/local_validation_targets.csv",
    ]
    if not all(path.exists() for path in required):
        subprocess.run([sys.executable, str(ROOT / "src/split_data.py")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tcn.yaml")
    parser.add_argument(
        "--full-training-from",
        type=Path,
        help="Local run directory whose config and best epoch should be retrained on train.csv",
    )
    args, overrides = parser.parse_known_args()
    if args.full_training_from:
        config = yaml.safe_load((args.full_training_from / "config.yaml").read_text())
        local_checkpoint = torch.load(
            args.full_training_from / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        config["run_name"] = f"{config['run_name']}_full_training"
        output_dir = ROOT / "output" / config["run_name"]
        train(
            config,
            pd.read_csv(ROOT / "res/dataset/train.csv"),
            validation=None,
            output_dir=output_dir,
            epochs=int(local_checkpoint["best_epoch"]),
        )
        future = pd.read_csv(ROOT / "res/dataset/validation_input.csv")
        index = pd.read_csv(ROOT / "res/dataset/forecast_index_validation.csv")
        checkpoint = torch.load(
            output_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        predictions = forecast(
            pd.read_csv(ROOT / "res/dataset/train.csv"), index, checkpoint, future
        )
        predictions.to_csv(
            output_dir
            / f"{config['run_name'].removesuffix('_full_training')}_validation.csv",
            index=False,
        )
        return

    config = load_config(args.config, overrides)
    ensure_local_split()
    output_dir = ROOT / "output" / config["run_name"]
    validation = (
        pd.read_csv(ROOT / "res/dataset/local_train.csv"),
        pd.read_csv(ROOT / "res/dataset/local_validation_input.csv"),
        pd.read_csv(ROOT / "res/dataset/local_validation_targets.csv"),
    )
    train(config, validation[0], validation=validation, output_dir=output_dir)


if __name__ == "__main__":
    main()
