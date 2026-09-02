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

Both commands train on `local_train.csv` and use the provided covariates; the TCN
also feeds the known future values directly to its forecast head. Both select the
best epoch by WAPE over the complete 336-hour local rollout and stop when validation
no longer improves.

Configuration values can be overridden without editing YAML:

```bash
uv run python src/train.py --config configs/tcn.yaml \
  run_name=tcn_large model.channels=64 max_epochs=500
```

Run names must be unique because an existing output directory is never silently
overwritten.

## Retrain a selected run on all public targets

After selecting a sensible local run, retrain its exact configuration for its
locally selected number of epochs:

```bash
uv run python src/train.py --full-training-from output/tcn
```

This creates `output/tcn_full_training/`, containing the resolved configuration,
checkpoint, and `tcn_validation.csv` ready for leaderboard evaluation. It does not
modify the `submission/` directory.
