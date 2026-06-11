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

To automate the entire local training split, baseline generation, metrics pre-computation, and dashboard visualization in a single run:

```bash
uv run python src/run_local_pipeline.py
```

This pipeline automatically handles the following sequence:
1. Splits `train.csv` to hold out the last 336 steps of each series for local validation.
2. Runs the baseline forecasts on the local training split.
3. Precomputes and writes model-specific metrics (e.g. `output/local_baselines/seasonal_mean_metrics.json`).
4. Launches the Streamlit evaluation dashboard.

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

#### On the Global Validation Index (for Hugging Face Leaderboard submission):
```bash
uv run res/provided_res/baseline/run_baselines.py \
  --train res/dataset/train.csv \
  --forecast-index res/dataset/forecast_index_validation.csv \
  --output-dir output/provided_baselines
```

#### On the Local Validation Split (for local backtesting):
```bash
uv run res/provided_res/baseline/run_baselines.py \
  --train res/dataset/local_train.csv \
  --forecast-index res/dataset/local_forecast_index_validation.csv \
  --output-dir output/local_baselines
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
