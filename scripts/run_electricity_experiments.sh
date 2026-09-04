#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

run_training() {
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

uv sync

for feature_set in raw operations_calendar calendar_extended; do
  run_training \
    configs/electricity_dlinear.yaml \
    "${feature_set}" \
    "output/electricity_load_diagrams/${feature_set}/dlinear"
  run_training \
    configs/electricity_tcn.yaml \
    "${feature_set}" \
    "output/electricity_load_diagrams/${feature_set}/tcn"
done

uv run python src/evaluate_predictions.py --dataset electricity

echo "Electricity experiments completed. Starting the dashboard."
exec uv run streamlit run src/dashboard.py
