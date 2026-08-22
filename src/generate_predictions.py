"""Development wrapper for running submission inference with explicit files."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
from predict import forecast  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--history", required=True, type=Path)
parser.add_argument("--forecast-index", required=True, type=Path)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
predictions = forecast(pd.read_csv(args.history), pd.read_csv(args.forecast_index), checkpoint)
args.output.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(args.output, index=False)
