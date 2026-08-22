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

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        return self.seasonal(x - trend) + self.trend(trend)


class CausalBlock(nn.Module):
    """Two residual dilated convolutions without access to future positions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.convolutions = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
        ])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for convolution in self.convolutions:
            x = convolution(x)[..., :-self.padding]
            x = self.dropout(self.activation(x))
        return x + residual


class TCN(nn.Module):
    """Global TCN with optional calendar features, series IDs, and residual forecast."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.prediction_length = config["prediction_length"]
        self.use_calendar = config["use_calendar"]
        self.use_series_embedding = config["use_series_embedding"]
        self.residual_period = config["residual_period"]
        channels = config["channels"]
        calendar_size = 4 if self.use_calendar else 0
        self.input_projection = nn.Conv1d(1 + calendar_size, channels, 1)
        self.blocks = nn.Sequential(*[
            CausalBlock(channels, config["kernel_size"], 2 ** index, config["dropout"])
            for index in range(config["levels"])
        ])
        embedding_size = config["embedding_size"] if self.use_series_embedding else 0
        if self.use_series_embedding:
            self.series_embedding = nn.Embedding(config["num_series"], embedding_size)
        self.head = nn.Sequential(
            nn.Linear(channels + embedding_size + calendar_size, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor, *, series_index: torch.Tensor | None = None,
                history_calendar: torch.Tensor | None = None,
                future_calendar: torch.Tensor | None = None) -> torch.Tensor:
        inputs = [x.unsqueeze(1)]
        if self.use_calendar:
            if history_calendar is None or future_calendar is None:
                raise ValueError("calendar-enabled TCN requires historical and future calendar features")
            inputs.append(history_calendar.transpose(1, 2))
        context = self.blocks(self.input_projection(torch.cat(inputs, dim=1)))[:, :, -1]
        context = context.unsqueeze(1).expand(-1, self.prediction_length, -1)
        head_inputs = [context]
        if self.use_series_embedding:
            if series_index is None:
                raise ValueError("series-embedding TCN requires series indices")
            embedding = self.series_embedding(series_index)
            head_inputs.append(embedding.unsqueeze(1).expand(-1, self.prediction_length, -1))
        if self.use_calendar:
            head_inputs.append(future_calendar)
        prediction = self.head(torch.cat(head_inputs, dim=-1)).squeeze(-1)
        if self.residual_period:
            prediction = prediction + x[:, -self.residual_period:][:, :self.prediction_length]
        return prediction


def build_model(config: dict) -> nn.Module:
    model_name = config.get("model", "dlinear")
    if model_name == "dlinear":
        return DLinear(
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            moving_average=config["moving_average"],
        )
    if model_name == "tcn":
        return TCN(config)
    raise ValueError(f"Unsupported submission model: {model_name!r}")
