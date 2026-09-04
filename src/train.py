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
sys.path.insert(0, str(ROOT))

from src.config import load_config, write_config  # noqa: E402
from src.data import (  # noqa: E402
    WindowDataset,
    fit_preprocessing,
    transformed_feature_count,
)
from src.datasets.registry import (  # noqa: E402
    base_table_path,
    canonical_dataset_name,
    model_category,
    model_run_dir,
    prepare_features,
    read_table,
    resolve_feature_set,
    split_table_path,
)
from src.evaluate_predictions import (  # noqa: E402
    calculate_metrics,
    calculate_overall_proxy,
    evaluate_file,
)
from src.model import build_model  # noqa: E402
from src.predict import forecast  # noqa: E402


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
    feature_columns: list[str],
    validation: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None,
    output_dir: Path,
    epochs: int | None = None,
) -> Path:
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    preprocessing = fit_preprocessing(train_frame, feature_columns)
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
    max_epochs = epochs if epochs is not None else config["max_epochs"]
    selection_metric = config["selection_metric"]
    best_score = float("inf")
    best_epoch = max_epochs
    stale_epochs = 0
    history_rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=False)
    write_config(config, output_dir / "config.yaml")
    checkpoint_path = output_dir / "checkpoint.pt"

    model_type = str(config["model"]["type"])
    model_label = {"dlinear": "DLinear", "tcn": "TCN"}.get(
        model_type.lower(), model_type
    )
    training_mode = "full training" if validation is None else "local holdout"
    covariate_names = ", ".join(feature_columns) if feature_columns else "none"
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print("\n" + "=" * 72)
    print(
        f"Starting {model_label} training | dataset={config['dataset']} "
        f"feature_set={config['feature_set']} run={config['run_name']}"
    )
    print(
        f"mode={training_mode} objective={config['training_objective']} "
        f"selection_metric={selection_metric} epochs={max_epochs} "
        f"early_stopping_patience={config['early_stopping_patience']}"
    )
    print(
        f"context={config['context_length']}h horizon={config['prediction_length']}h "
        f"stride={config['stride']} batch_size={config['batch_size']}"
    )
    print(
        f"series={len(series_mapping)} windows={len(dataset):,} "
        f"future_covariates={len(feature_columns)} [{covariate_names}]"
    )
    print(
        f"device={device} trainable_parameters={trainable_parameters:,} "
        f"learning_rate={config['learning_rate']}"
    )
    print(f"output={output_dir}")
    print("=" * 72, flush=True)

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
            # Approximate the public "Overall" leaderboard, which percentile-ranks MAE,
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
            wape_loss = absolute_error.sum() / raw_target.abs().sum().clamp_min(epsilon)
            if config["training_objective"] == "wape":
                # Electricity contains legitimate zeros, making pointwise MAPE an
                # unsuitable optimization target. WAPE remains stable at zeros.
                loss = wape_loss
            else:
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
                        wape_loss,
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
        train_overall = total_loss / len(dataset)
        train_mae = total_mae / len(dataset)
        print(
            f"epoch={epoch:03d} train_{config['training_objective']}="
            f"{train_overall:.6f} "
            f"train_mae={train_mae:.6f}"
        )
        history_row = {
            "epoch": epoch,
            "training_objective": config["training_objective"],
            "train_loss": train_overall,
            "train_mae": train_mae,
        }

        if validation is None:
            history_rows.append(history_row)
            pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
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
        history_row.update(
            {
                "validation_overall_proxy": overall_proxy,
                "validation_mae": scores["MAE"],
                "validation_mse": scores["MSE"],
                "validation_rmse": scores["RMSE"],
                "validation_mape_pct": scores["MAPE (%)"],
                "validation_smape_pct": scores["sMAPE (%)"],
                "validation_wape_pct": scores["WAPE"],
            }
        )
        should_stop = False
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
                should_stop = True
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
        if should_stop:
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
            f"{model_category(config['dataset'], config['feature_set'])} / "
            f"{config['run_name']}",
            best_epoch=best_epoch,
            category=model_category(config["dataset"], config["feature_set"]),
            name=config["run_name"],
        )
    return checkpoint_path


def ensure_local_split(dataset: str, horizon: int) -> None:
    required = [
        split_table_path(dataset, horizon, "local_train"),
        split_table_path(dataset, horizon, "local_validation_input"),
        split_table_path(dataset, horizon, "local_validation_targets"),
    ]
    if not all(path.exists() for path in required):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.datasets.split_data",
                "--dataset",
                dataset,
                "--horizon",
                str(horizon),
            ],
            check=True,
        )


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
        config.setdefault("dataset", "operations_forecasting_2026")
        config.setdefault("feature_set", "provided")
        config.setdefault("training_objective", "overall")
        dataset_name = canonical_dataset_name(config["dataset"])
        feature_set = resolve_feature_set(dataset_name, config["feature_set"])
        config["dataset"] = dataset_name
        config["feature_set"] = feature_set
        local_checkpoint = torch.load(
            args.full_training_from / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        config["run_name"] = f"{config['run_name']}_full_training"
        output_dir = model_run_dir(dataset_name, feature_set, config["run_name"])
        full_train, columns = prepare_features(
            read_table(base_table_path(dataset_name, "train")),
            dataset_name,
            feature_set,
        )
        train(
            config,
            full_train,
            feature_columns=columns,
            validation=None,
            output_dir=output_dir,
            epochs=int(local_checkpoint["best_epoch"]),
        )
        future_path = base_table_path(dataset_name, "validation_input")
        index_path = base_table_path(dataset_name, "forecast_index_validation")
        if not future_path.exists() or not index_path.exists():
            print(
                f"Wrote full-training checkpoint to {output_dir}. "
                f"{dataset_name} has no external validation input/index, so no "
                "public-validation prediction file was generated."
            )
            return
        future, _ = prepare_features(
            read_table(future_path), dataset_name, feature_set
        )
        index = read_table(index_path)
        checkpoint = torch.load(
            output_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        predictions = forecast(
            full_train, index, checkpoint, future
        )
        predictions.to_csv(
            output_dir
            / f"{config['run_name'].removesuffix('_full_training')}_validation.csv",
            index=False,
        )
        return

    config = load_config(args.config, overrides)
    dataset_name = canonical_dataset_name(config["dataset"])
    feature_set = resolve_feature_set(dataset_name, config["feature_set"])
    config["dataset"] = dataset_name
    config["feature_set"] = feature_set
    horizon = int(config["prediction_length"])
    ensure_local_split(dataset_name, horizon)
    output_dir = model_run_dir(dataset_name, feature_set, config["run_name"])
    local_train, columns = prepare_features(
        read_table(split_table_path(dataset_name, horizon, "local_train")),
        dataset_name,
        feature_set,
    )
    local_future, _ = prepare_features(
        read_table(split_table_path(dataset_name, horizon, "local_validation_input")),
        dataset_name,
        feature_set,
    )
    validation = (
        local_train,
        local_future,
        read_table(split_table_path(dataset_name, horizon, "local_validation_targets")),
    )
    train(
        config,
        validation[0],
        feature_columns=columns,
        validation=validation,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
