# How to Train and Evaluate

The commands below create the local split automatically when it is missing. Neural
runs write a resolved `config.yaml`, their best `checkpoint.pt`, local
`predictions.csv`, and `metrics.json` to `output/<run_name>/`.

## Run all local baselines

```bash
uv run python src/run_baselines.py
```

This generates and evaluates naive-last-value, lag-24, lag-168, and seasonal-mean
forecasts in `output/local_baselines/`.

## Train multivariate DLinear

```bash
uv run python src/train.py --config configs/dlinear.yaml
```

## Train multivariate TCN

```bash
uv run python src/train.py --config configs/tcn.yaml
```

Both commands train on `local_train.csv`, use the known future covariates, and
predict the complete 336-hour horizon in one pass. No predicted target is reused as
model input. Both select the best epoch with a scale-free local proxy for the
public Overall rank on the exact 336-hour holdout. The training objective combines
MAE, MSE, RMSE, MAPE, sMAPE, and WAPE in the original target scale.

Configuration values can be overridden without editing YAML:

```bash
uv run python src/train.py --config configs/tcn.yaml \
  run_name=tcn_large model.channels=96 max_epochs=300
```

Run names must be unique because an existing output directory is never silently
overwritten.

## Retrain a selected run on all public targets

After selecting a sensible local run, retrain its exact configuration for its
locally selected number of epochs:

```bash
uv run python src/train.py --full-training-from output/tcn_direct336
```

This creates `output/tcn_direct336_full_training/`, containing the resolved
configuration, checkpoint, and `tcn_direct336_validation.csv` ready for public
leaderboard evaluation. It does not modify the `submission/` directory.
