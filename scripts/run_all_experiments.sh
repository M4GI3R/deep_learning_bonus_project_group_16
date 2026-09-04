#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

run_local_training() {
  local config_path="$1"
  local feature_set="$2"
  local output_path="$3"

  if [[ -f "${output_path}/checkpoint.pt" && -f "${output_path}/metrics.json" ]]; then
    echo "Skipping completed run: ${output_path}"
    return
  fi
  if [[ -e "${output_path}" ]]; then
    echo "Incomplete output directory already exists: ${output_path}" >&2
    echo "Rename or remove it before restarting this workflow." >&2
    exit 1
  fi
  uv run python src/train.py --config "${config_path}" "feature_set=${feature_set}"
}

run_full_training() {
  local selected_run="$1"
  local output_path="$2"
  local model_name
  model_name="$(basename -- "${selected_run}")"
  local validation_path="${output_path}/${model_name}_validation.csv"

  if [[ -f "${output_path}/checkpoint.pt" && -f "${validation_path}" ]]; then
    echo "Skipping completed full-training run: ${output_path}"
    return
  fi
  if [[ -e "${output_path}" ]]; then
    echo "Incomplete output directory already exists: ${output_path}" >&2
    echo "Rename or remove it before restarting this workflow." >&2
    exit 1
  fi
  uv run python src/train.py --full-training-from "${selected_run}"
}

uv sync
uv run python -m src.datasets.import_dataset operations
uv run python -m src.datasets.import_dataset electricity

uv run python src/run_baselines.py --dataset operations
run_local_training \
  configs/dlinear.yaml \
  provided \
  output/operations_forecasting_2026/dlinear
run_local_training \
  configs/tcn.yaml \
  provided \
  output/operations_forecasting_2026/tcn
run_full_training \
  output/operations_forecasting_2026/dlinear \
  output/operations_forecasting_2026/dlinear_full_training
run_full_training \
  output/operations_forecasting_2026/tcn \
  output/operations_forecasting_2026/tcn_full_training

uv run python src/run_baselines.py --dataset electricity
for feature_set in raw operations_calendar calendar_extended; do
  run_local_training \
    configs/electricity_dlinear.yaml \
    "${feature_set}" \
    "output/electricity_load_diagrams/${feature_set}/dlinear"
  run_local_training \
    configs/electricity_tcn.yaml \
    "${feature_set}" \
    "output/electricity_load_diagrams/${feature_set}/tcn"
done

uv run python src/evaluate_predictions.py --dataset operations
uv run python src/evaluate_predictions.py --dataset electricity

echo "All experiments completed. Starting the dashboard."
exec uv run streamlit run src/dashboard.py
