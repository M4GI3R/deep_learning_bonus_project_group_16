# Deep Learning Bonus Project - Group 16

This repository contains the codebase for the Deep Learning Bonus Project by Group 16. The project uses `uv` for python environment and dependency management.

---

## WSL (Windows Subsystem for Linux) Setup Guide

Follow the steps below in your WSL terminal to configure the environment and download the dataset.

### Prerequisites

Ensure you have `uv` installed in WSL. If not, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Synchronize Dependencies

Run `uv sync` to configure the virtual environment and install required libraries:
```bash
uv sync
```

### 2. Activate the Virtual Environment

Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 3. Download the Dataset

Download the dataset `AIML-TUDA/dlam-ts-project-data-2026` from Hugging Face into `res/dataset/`:
```bash
uv run --with huggingface_hub src/download_data.py
```

---

## Local Evaluation Pipeline (Recommended)

Use these three commands for the complete local split, baseline generation, neural
training, rollout prediction, and metric computation:

```bash
uv run python src/run_baselines.py
uv run python src/train.py --config configs/dlinear.yaml
uv run python src/train.py --config configs/tcn.yaml
```

Each command creates the local split when needed and writes predictions and metrics.
Neural runs additionally save the resolved YAML configuration and best checkpoint
under `output/<run_name>/`; validation WAPE over the complete rollout controls early
stopping.

---

## Running Individual Steps

If you want to run the pipeline steps individually or customize the arguments:

### A. Split Training Data
Create a local hold-out validation set (last 336 hours per series) from `train.csv`:
```bash
uv run python src/split_data.py
```
This generates `local_train.csv`, `local_validation_input.csv`, `local_validation_targets.csv`, and `local_forecast_index_validation.csv` inside `res/dataset/`.

### B. Run Baseline Forecasts

#### On the local validation split:
```bash
uv run python src/run_baselines.py
```

### C. Precompute Performance Metrics
Compute model accuracy metrics (MAE, RMSE, sMAPE, WAPE, and improvement vs. naive baseline) across all units and steps:
```bash
uv run python src/evaluate_predictions.py
```
This scans `output/` recursively for prediction files and saves individual `[model]_metrics.json` summaries in each model's respective folder.

### D. Launch the Streamlit Dashboard
Launch the interface to compare configurations, WAPE improvement, error distribution, and rollout drift:
```bash
uv run python dashboard.py
```
*(Alternatively, run `uv run streamlit run src/dashboard.py`)*

---

## Multivariate neural models

Train multivariate DLinear on the local split:

```bash
uv run python src/train.py --config configs/dlinear.yaml
```

Train the multivariate TCN:

```bash
uv run python src/train.py --config configs/tcn.yaml
```

Both models consume the released historical covariates; the TCN additionally uses
the known future values directly. Missing numerical covariates are set to the
training mean after standardization. Override settings without changing YAML:

```bash
uv run python src/train.py --config configs/tcn.yaml \
  run_name=tcn_large model.channels=64
```

Retrain a selected configuration on all public targets for the locally selected
number of epochs and generate its validation CSV:

```bash
uv run python src/train.py --full-training-from output/tcn
```

See `docs/how_to_train_and_submit.md` for the generated files and full workflow.
