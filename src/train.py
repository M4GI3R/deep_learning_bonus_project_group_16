"""Small training entrypoint for the DLinear proof model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
from src.model import build_model  # noqa: E402


class WindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, context: int, horizon: int, stride: int,
                 statistics: dict[str, tuple[float, float]]) -> None:
        self.context, self.horizon = context, horizon
        self.series = {}
        self.windows = []
        for series_id, part in frame.groupby("series_id", sort=False):
            values = part.sort_values("timestamp")["target"].to_numpy(np.float32)
            mean, scale = statistics[str(series_id)]
            self.series[str(series_id)] = (values - mean) / scale
            for start in range(0, len(values) - context - horizon + 1, stride):
                self.windows.append((str(series_id), start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_id, start = self.windows[index]
        values = self.series[series_id]
        split = start + self.context
        return torch.from_numpy(values[start:split]), torch.from_numpy(values[split:split + self.horizon])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact DLinear proof model")
    parser.add_argument("--train", type=Path, default=ROOT / "res/dataset/local_train.csv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "submission/checkpoint.pt")
    parser.add_argument("--context-length", type=int, default=336)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--moving-average", type=int, default=25)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = pd.read_csv(args.train).sort_values(["series_id", "timestamp"])
    required = {"series_id", "timestamp", "target"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError(f"Training data must contain complete columns {sorted(required)}")

    statistics = {}
    for series_id, part in frame.groupby("series_id", sort=False):
        values = part["target"].to_numpy(float)
        scale = float(values.std())
        statistics[str(series_id)] = (float(values.mean()), scale if scale > 1e-6 else 1.0)

    config = {"model": "dlinear", "context_length": args.context_length,
              "prediction_length": args.prediction_length, "moving_average": args.moving_average}
    dataset = WindowDataset(frame, args.context_length, args.prediction_length, args.stride, statistics)
    if not len(dataset):
        raise ValueError("Training data is too short for the requested context and prediction lengths")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"epoch={epoch:03d} mae_normalized={total / len(dataset):.6f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "series_statistics": statistics,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
