"""Direct multi-horizon DLinear and TCN forecasting models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("moving-average kernel size must be a positive odd number")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        padded = torch.cat(
            [x[:, :1].repeat(1, padding, 1), x, x[:, -1:].repeat(1, padding, 1)], dim=1
        )
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class DLinear(nn.Module):
    """DLinear/NLinear hybrid with a known-future covariate head."""

    def __init__(
        self,
        context_length: int = 336,
        prediction_length: int = 336,
        moving_average: int = 25,
        num_features: int = 0,
        num_series: int = 1,
        embedding_size: int = 0,
        exogenous_hidden_size: int = 64,
        dropout: float = 0.0,
        use_series_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.moving_average = MovingAverage(moving_average)
        self.seasonal = nn.Linear(context_length, prediction_length)
        self.trend = nn.Linear(context_length, prediction_length)
        self.use_series_embedding = use_series_embedding

        nn.init.constant_(self.seasonal.weight, 1.0 / context_length)
        nn.init.constant_(self.trend.weight, 1.0 / context_length)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)

        embedding_input_size = 0
        if use_series_embedding:
            self.series_embedding = nn.Embedding(num_series, embedding_size)
            embedding_input_size = embedding_size

        self.future_head = nn.Sequential(
            nn.Linear(num_features + embedding_input_size, exogenous_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(exogenous_hidden_size, 1),
        )
        nn.init.zeros_(self.future_head[-1].weight)
        nn.init.zeros_(self.future_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        series_index: torch.Tensor | None = None,
        history_features: torch.Tensor | None = None,
        future_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del history_features
        if future_features is None:
            raise ValueError("DLinear requires known future features")

        level = x[:, -1:]
        centered = x - level
        trend = self.moving_average(centered).squeeze(-1)
        seasonal = centered - trend
        target_forecast = self.seasonal(seasonal) + self.trend(trend) + level

        head_inputs = [future_features]
        if self.use_series_embedding:
            if series_index is None:
                raise ValueError("series-embedding DLinear requires series indices")
            embedding = self.series_embedding(series_index)
            head_inputs.append(
                embedding.unsqueeze(1).expand(-1, future_features.shape[1], -1)
            )
        exogenous_correction = self.future_head(
            torch.cat(head_inputs, dim=-1)
        ).squeeze(-1)
        return target_forecast + exogenous_correction


class CausalBlock(nn.Module):
    """Residual block with two dilated causal convolutions."""

    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        convolutions = []
        for _ in range(2):
            convolution = nn.Conv1d(
                channels, channels, kernel_size, padding=padding, dilation=dilation
            )
            nn.init.normal_(convolution.weight, mean=0.0, std=0.01)
            nn.init.zeros_(convolution.bias)
            convolutions.append(weight_norm(convolution))
        self.convolutions = nn.ModuleList(convolutions)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for convolution in self.convolutions:
            x = convolution(x)
            if self.padding:
                x = x[..., : -self.padding]
            x = self.dropout(self.activation(x))
        return self.activation(x + residual)


class TCN(nn.Module):
    """Direct sequence-to-sequence TCN over history and future covariates."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.prediction_length = config["prediction_length"]
        self.use_series_embedding = config["use_series_embedding"]
        self.residual_period = config["residual_period"]
        channels = config["channels"]
        feature_size = config["num_features"]
        self.input_projection = nn.Conv1d(2 + feature_size, channels, 1)
        self.blocks = nn.Sequential(
            *[
                CausalBlock(
                    channels, config["kernel_size"], 2**index, config["dropout"]
                )
                for index in range(config["levels"])
            ]
        )
        embedding_size = config["embedding_size"] if self.use_series_embedding else 0
        if self.use_series_embedding:
            self.series_embedding = nn.Embedding(config["num_series"], embedding_size)
        self.head = nn.Sequential(
            nn.Linear(channels + embedding_size, channels),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(channels, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        series_index: torch.Tensor | None = None,
        history_features: torch.Tensor | None = None,
        future_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_features is None or future_features is None:
            raise ValueError("multivariate TCN requires historical and future features")
        if future_features.shape[1] != self.prediction_length:
            raise ValueError(
                "future feature length must equal the configured prediction length"
            )

        period = min(self.residual_period, x.shape[1])
        seasonal_history = x[:, -period:]
        repetitions = (self.prediction_length + period - 1) // period
        seasonal_baseline = seasonal_history.repeat(1, repetitions)[
            :, : self.prediction_length
        ]

        target_path = torch.cat([x, seasonal_baseline], dim=1).unsqueeze(-1)
        feature_path = torch.cat([history_features, future_features], dim=1)
        future_indicator = torch.cat(
            [
                x.new_zeros((x.shape[0], x.shape[1], 1)),
                x.new_ones((x.shape[0], self.prediction_length, 1)),
            ],
            dim=1,
        )
        sequence = torch.cat(
            [target_path, feature_path, future_indicator], dim=-1
        ).transpose(1, 2)
        encoded_future = self.blocks(self.input_projection(sequence))[
            :, :, -self.prediction_length :
        ].transpose(1, 2)

        head_inputs = [encoded_future]
        if self.use_series_embedding:
            if series_index is None:
                raise ValueError("series-embedding TCN requires series indices")
            embedding = self.series_embedding(series_index)
            head_inputs.append(
                embedding.unsqueeze(1).expand(-1, self.prediction_length, -1)
            )
        correction = self.head(torch.cat(head_inputs, dim=-1)).squeeze(-1)
        return seasonal_baseline + correction


def build_model(config: dict) -> nn.Module:
    model_name = config.get("model", "dlinear")
    if model_name == "dlinear":
        return DLinear(
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            moving_average=config["moving_average"],
            num_features=config["num_features"],
            num_series=config["num_series"],
            embedding_size=config.get("embedding_size", 0),
            exogenous_hidden_size=config.get("exogenous_hidden_size", 64),
            dropout=config.get("dropout", 0.0),
            use_series_embedding=config.get("use_series_embedding", False),
        )
    if model_name == "tcn":
        return TCN(config)
    raise ValueError(f"Unsupported model: {model_name!r}")
