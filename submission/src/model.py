"""Fixed TCN architecture used for final private evaluation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


CONTEXT_LENGTH = 168
PREDICTION_LENGTH = 336
CHANNELS = 64
LEVELS = 7
KERNEL_SIZE = 3
DROPOUT = 0.15
EMBEDDING_SIZE = 16
RESIDUAL_PERIOD = 24


class CausalBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        padding = (KERNEL_SIZE - 1) * dilation
        self.padding = padding
        convolutions = []
        for _ in range(2):
            convolution = nn.Conv1d(
                CHANNELS,
                CHANNELS,
                KERNEL_SIZE,
                padding=padding,
                dilation=dilation,
            )
            convolutions.append(weight_norm(convolution))
        self.convolutions = nn.ModuleList(convolutions)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout1d(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for convolution in self.convolutions:
            x = convolution(x)
            if self.padding:
                x = x[..., : -self.padding]
            x = self.dropout(self.activation(x))
        return self.activation(x + residual)


class FinalTCN(nn.Module):
    """The fixed operations TCN selected by the project experiments."""

    def __init__(self, *, num_features: int, num_series: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(2 + num_features, CHANNELS, 1)
        self.blocks = nn.Sequential(
            *[CausalBlock(dilation=2**index) for index in range(LEVELS)]
        )
        self.series_embedding = nn.Embedding(num_series, EMBEDDING_SIZE)
        self.head = nn.Sequential(
            nn.Linear(CHANNELS + EMBEDDING_SIZE, CHANNELS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(CHANNELS, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        series_index: torch.Tensor,
        history_features: torch.Tensor,
        future_features: torch.Tensor,
    ) -> torch.Tensor:
        seasonal_history = x[:, -RESIDUAL_PERIOD:]
        repetitions = (PREDICTION_LENGTH + RESIDUAL_PERIOD - 1) // RESIDUAL_PERIOD
        seasonal_baseline = seasonal_history.repeat(1, repetitions)[
            :, :PREDICTION_LENGTH
        ]

        target_path = torch.cat([x, seasonal_baseline], dim=1).unsqueeze(-1)
        feature_path = torch.cat([history_features, future_features], dim=1)
        future_indicator = torch.cat(
            [
                x.new_zeros((x.shape[0], CONTEXT_LENGTH, 1)),
                x.new_ones((x.shape[0], PREDICTION_LENGTH, 1)),
            ],
            dim=1,
        )
        sequence = torch.cat(
            [target_path, feature_path, future_indicator], dim=-1
        ).transpose(1, 2)
        encoded_future = self.blocks(self.input_projection(sequence))[
            :, :, -PREDICTION_LENGTH:
        ].transpose(1, 2)

        embedding = self.series_embedding(series_index)
        embedded_series = embedding.unsqueeze(1).expand(
            -1, PREDICTION_LENGTH, -1
        )
        correction = self.head(
            torch.cat([encoded_future, embedded_series], dim=-1)
        ).squeeze(-1)
        return seasonal_baseline + correction
