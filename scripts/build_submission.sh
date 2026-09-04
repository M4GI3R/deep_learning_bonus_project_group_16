#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SOURCE_CHECKPOINT="output/operations_forecasting_2026/tcn_full_training/checkpoint.pt"
SUBMISSION_DIR="submission"
INPUT_DIR="res/datasets/operations_forecasting_2026/processed"
SMOKE_OUTPUT="tmp/submission_validation_predictions.csv"

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing final TCN checkpoint: ${SOURCE_CHECKPOINT}" >&2
  echo "Run the operations TCN full-training command first." >&2
  exit 1
fi

for required_input in train.csv validation_input.csv forecast_index_validation.csv; do
  if [[ ! -f "${INPUT_DIR}/${required_input}" ]]; then
    echo "Missing validation input: ${INPUT_DIR}/${required_input}" >&2
    echo "Import the operations dataset before building the archive." >&2
    exit 1
  fi
done

rm -f "${SUBMISSION_DIR}/final_submission.zip"
cp "${SOURCE_CHECKPOINT}" "${SUBMISSION_DIR}/checkpoint.pt"
mkdir -p tmp

# Exercise the same standalone entry point used by private evaluation before
# packaging it. The specialized loader rejects any non-final-TCN checkpoint.
uv run python "${SUBMISSION_DIR}/predict.py" \
  --input_dir "${INPUT_DIR}" \
  --output_file "${SMOKE_OUTPUT}" \
  --checkpoint "${SUBMISSION_DIR}/checkpoint.pt"

(
  cd "${SUBMISSION_DIR}"
  if command -v zip >/dev/null 2>&1; then
    zip -q final_submission.zip \
      predict.py requirements.txt checkpoint.pt src/__init__.py src/model.py
  else
    uv run python -c 'from zipfile import ZIP_DEFLATED, ZipFile; files = ("predict.py", "requirements.txt", "checkpoint.pt", "src/__init__.py", "src/model.py"); archive = ZipFile("final_submission.zip", "w", ZIP_DEFLATED); [archive.write(path, path) for path in files]; archive.close()'
  fi
)

echo "Validated TCN submission archive: ${SUBMISSION_DIR}/final_submission.zip"
echo "Archive contents:"
if command -v unzip >/dev/null 2>&1; then
  unzip -Z1 "${SUBMISSION_DIR}/final_submission.zip"
else
  uv run python -m zipfile -l "${SUBMISSION_DIR}/final_submission.zip"
fi
