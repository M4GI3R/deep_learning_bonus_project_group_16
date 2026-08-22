"""Compact DLinear model used for the first learned forecasting submission."""

from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel size must be a positive odd number")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        padded = torch.cat(
            [x[:, :1].repeat(1, padding), x, x[:, -1:].repeat(1, padding)], dim=1
        )
        return self.pool(padded.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    """Decompose a univariate context and project trend/seasonality directly."""

    def __init__(self, context_length: int = 336, prediction_length: int = 24,
                 moving_average: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(moving_average)
        self.seasonal = nn.Linear(context_length, prediction_length)
        self.trend = nn.Linear(context_length, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        return self.seasonal(x - trend) + self.trend(trend)


def build_model(config: dict) -> nn.Module:
    if config.get("model", "dlinear") != "dlinear":
        raise ValueError(f"Unsupported submission model: {config.get('model')!r}")
    return DLinear(
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        moving_average=config["moving_average"],
    )
