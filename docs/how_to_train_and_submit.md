# Training, evaluation, and submission

This guide describes the lifecycle of a run: local model selection, metric
refresh, full-data retraining, and operations-benchmark packaging. Run all
commands from the repository root in Linux or WSL.

## 1. Prepare the environment and data

```bash
uv sync
uv run python -m src.datasets.import_dataset operations
uv run python -m src.datasets.import_dataset electricity
```

The importers are idempotent by default. Dataset manifests in
`configs/datasets/` define canonical names, aliases, storage formats, default
horizons, and feature sets. Local chronological splits are generated
automatically when a baseline or model first needs them.

## 2. Run local baselines

```bash
uv run python src/run_baselines.py --dataset operations
uv run python src/run_baselines.py --dataset electricity
```

Both commands use the final 336 hours of every series as the holdout and write
naive-last-value, lag-24, lag-168, and seasonal-mean predictions and metrics below
the selected dataset's `baselines/horizon_336/` folder.

## 3. Train local neural models

Operations benchmark:

```bash
uv run python src/train.py --config configs/dlinear.yaml
uv run python src/train.py --config configs/tcn.yaml
```

Electricity ablation:

```bash
uv run python src/train.py --config configs/electricity_dlinear.yaml feature_set=raw
uv run python src/train.py --config configs/electricity_tcn.yaml feature_set=raw

uv run python src/train.py --config configs/electricity_dlinear.yaml feature_set=operations_calendar
uv run python src/train.py --config configs/electricity_tcn.yaml feature_set=operations_calendar

uv run python src/train.py --config configs/electricity_dlinear.yaml feature_set=calendar_extended
uv run python src/train.py --config configs/electricity_tcn.yaml feature_set=calendar_extended
```

Every model predicts all 336 hours directly. Operations training optimizes and
selects on the scale-free Overall proxy defined by the six benchmark metrics.
Electricity training optimizes and selects on WAPE because its legitimate zero
loads make pointwise MAPE unsuitable as an objective.

A local neural run contains:

```text
config.yaml       Fully resolved configuration
checkpoint.pt     Best local checkpoint
history.csv       Epoch-level training and validation history
predictions.csv   Forecasts for the local holdout
metrics.json      Dashboard-ready local metrics
```

Output directories must be unique. Override settings on the command line when a
separate run is intended:

```bash
uv run python src/train.py --config configs/tcn.yaml \
  run_name=tcn_large model.channels=96 max_epochs=300
```

## 4. Refresh local metrics

```bash
uv run python src/evaluate_predictions.py --dataset operations
uv run python src/evaluate_predictions.py --dataset electricity
```

Evaluation is dataset-local. MAE, MSE, RMSE, MAPE, sMAPE, and WAPE are computed
in the selected dataset's native target scale; no cross-dataset normalization is
applied. MAPE excludes observations whose actual target is zero. Full-training
public-validation CSVs are ignored because their hidden labels are not the local
holdout labels.

Inspect results with:

```bash
uv run streamlit run src/dashboard.py
```

## 5. Retrain an operations model on all available targets

After choosing a local checkpoint, retrain its resolved configuration from a new
random initialization on all available operations observations. The number of
epochs is taken from the selected local checkpoint's best epoch.

```bash
uv run python src/train.py --full-training-from output/operations_forecasting_2026/dlinear
uv run python src/train.py --full-training-from output/operations_forecasting_2026/tcn
```

The commands create `dlinear_full_training/` and `tcn_full_training/` beside the
local runs. Each folder contains a full-data checkpoint and the corresponding
`dlinear_validation.csv` or `tcn_validation.csv` for public validation. Local
evaluation intentionally does not score these files.

## 6. Build the final TCN archive

The code repository and final model archive are separate deliverables. Repository
training and evaluation use `src/model.py` and `src/predict.py`; no repository
command imports code from `submission/`.

The submission folder contains a deployment-only copy of the selected operations
TCN. Its architecture is fixed to the final configuration: 168 context steps,
336 output steps, 64 channels, seven residual blocks, kernel size 3, dropout 0.15,
16-dimensional series embeddings, and a 24-hour residual baseline. It contains
no DLinear implementation or model-selection options.

After producing `output/operations_forecasting_2026/tcn_full_training/checkpoint.pt`,
run:

```bash
bash scripts/build_submission.sh
```

The builder:

1. copies the final TCN checkpoint to `submission/checkpoint.pt`;
2. executes the standalone submission entry point against the public validation
   inputs;
3. rejects an incompatible checkpoint; and
4. creates `submission/final_submission.zip` without caches or repository files.

It uses the WSL `zip` utility when available and otherwise falls back to Python's
standard-library ZIP implementation, so no additional system package is required.

The archive root contains only the required runtime files:

```text
predict.py
requirements.txt
checkpoint.pt
src/
  __init__.py
  model.py
```

Private evaluation can therefore extract the archive into `/submission` and run
the assignment's exact command:

```bash
python predict.py \
  --input_dir /data/input \
  --output_file /output/predictions.csv \
  --checkpoint /submission/checkpoint.pt
```

`predict.py` prefers `forecast_index_test.csv` and `test_input.csv`, falling back
to their validation counterparts only for the local smoke test. It writes exactly
one finite `prediction` for every requested `(series_id, timestamp)` pair and
preserves forecast-index row order.

The repository intentionally ignores `submission/checkpoint.pt` and the generated
ZIP because both are derived artifacts.

## Automated workflows

For a restart-safe sequential run of the full project:

```bash
bash scripts/run_all_experiments.sh
```

For the six electricity model runs only, assuming its dataset and baselines are
already present:

```bash
bash scripts/run_electricity_experiments.sh
```
