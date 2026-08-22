# How to Train and Submit

Run the following on the GPU.

## Train and evaluate locally

```bash
# Ensure the local split exists
uv run python src/split_data.py

# Train on the local split
uv run python src/train.py

# Generate local predictions
uv run python src/generate_predictions.py \
  --history res/dataset/local_train.csv \
  --forecast-index res/dataset/local_forecast_index_validation.csv \
  --output output/local_dlinear/predictions.csv \
  --checkpoint submission/checkpoint.pt

# Evaluate locally
uv run python src/evaluate_predictions.py
```

## Retrain on all public training data

If the local result is sensible, retrain using all public training targets:

```bash
uv run python src/train.py \
  --train res/dataset/train.csv \
  --checkpoint submission/checkpoint.pt
```

## Generate and upload the validation submission

Generate the Hugging Face validation file:

```bash
uv run python submission/predict.py \
  --input_dir res/dataset \
  --output_file output/dlinear_validation.csv \
  --checkpoint submission/checkpoint.pt
```

Upload `output/dlinear_validation.csv` to the public validation leaderboard.
