import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "submission"))

from config import load_config
from data import (
    FEATURE_COLUMNS,
    WindowDataset,
    fit_preprocessing,
    transform_features,
    transformed_feature_count,
)
from predict import forecast
from evaluate_predictions import calculate_metrics, calculate_overall_proxy
from src.model import build_model


def frame_for(
    hours: int, *, with_target: bool = True, start: str = "2023-01-01"
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=hours, freq="h")
    frame = pd.DataFrame({"series_id": "unit_000", "timestamp": timestamps})
    for index, column in enumerate(FEATURE_COLUMNS):
        frame[column] = np.sin(np.arange(hours) / (index + 2))
    frame.loc[2, "queue_pressure_forecast"] = np.nan
    if with_target:
        frame["target"] = 2 + np.sin(np.arange(hours) * 2 * np.pi / 24)
    return frame


class PipelineTests(unittest.TestCase):
    def test_metrics_use_public_leaderboard_scale(self):
        y_true = np.array([1.0, 2.0, 4.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["MAE"], 1.0)
        self.assertAlmostEqual(metrics["MSE"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(5.0 / 3.0))
        self.assertAlmostEqual(metrics["MAPE (%)"], 50.0)
        self.assertAlmostEqual(metrics["WAPE"], 300.0 / 7.0)
        self.assertTrue(np.isfinite(calculate_overall_proxy(y_true, y_pred)))

    def test_config_overrides_nested_values(self):
        config = load_config(
            ROOT / "configs/tcn.yaml", ["run_name=test", "model.channels=64"]
        )
        self.assertEqual(config["run_name"], "test")
        self.assertEqual(config["model"]["channels"], 64)

    def test_preprocessing_imputes_released_covariate_nans(self):
        frame = frame_for(200)
        preprocessing = fit_preprocessing(frame)
        transformed = transform_features(frame, preprocessing)
        self.assertEqual(transformed.shape, (200, len(FEATURE_COLUMNS) + 1))
        self.assertTrue(np.isfinite(transformed).all())

    def test_multivariate_window_and_models(self):
        frame = frame_for(520)
        preprocessing = fit_preprocessing(frame)
        dataset = WindowDataset(frame, 168, 336, 12, preprocessing, {"unit_000": 0})
        (
            x,
            _y,
            series,
            history_features,
            future_features,
            target_mean,
            target_scale,
        ) = dataset[0]
        common = {
            "context_length": 168,
            "prediction_length": 336,
            "moving_average": 25,
            "num_features": transformed_feature_count(preprocessing),
            "num_series": 1,
        }
        dlinear = build_model(
            {
                **common,
                "model": "dlinear",
                "embedding_size": 2,
                "exogenous_hidden_size": 8,
                "dropout": 0.0,
                "use_series_embedding": True,
            }
        )
        output = dlinear(
            x[None],
            series_index=series[None],
            history_features=history_features[None],
            future_features=future_features[None],
        )
        self.assertEqual(output.shape, (1, 336))
        self.assertGreater(float(target_mean), 0.0)
        self.assertGreater(float(target_scale), 0.0)
        tcn = build_model(
            {
                **common,
                "model": "tcn",
                "channels": 4,
                "levels": 2,
                "kernel_size": 3,
                "dropout": 0.0,
                "embedding_size": 2,
                "residual_period": 24,
                "use_series_embedding": True,
            }
        )
        output = tcn(
            x[None],
            series_index=series[None],
            history_features=history_features[None],
            future_features=future_features[None],
        )
        self.assertEqual(output.shape, (1, 336))

    def test_forecast_covers_complete_index_in_one_direct_pass(self):
        history = frame_for(200)
        future = frame_for(336, with_target=False, start="2023-01-09 08:00:00")
        preprocessing = fit_preprocessing(history)
        config = {
            "model": "dlinear",
            "context_length": 168,
            "prediction_length": 336,
            "moving_average": 25,
            "num_features": transformed_feature_count(preprocessing),
            "num_series": 1,
            "embedding_size": 2,
            "exogenous_hidden_size": 8,
            "dropout": 0.0,
            "use_series_embedding": True,
            "prediction_floor": 0.0,
        }
        model = build_model(config)
        checkpoint = {
            "config": config,
            "preprocessing": preprocessing,
            "series_mapping": {"unit_000": 0},
            "state_dict": model.state_dict(),
        }
        result = forecast(
            history, future[["series_id", "timestamp"]], checkpoint, future
        )
        self.assertEqual(len(result), 336)
        self.assertTrue(np.isfinite(result["prediction"]).all())
        self.assertTrue(result["prediction"].ge(0.0).all())

    def test_forecast_rejects_autoregressive_target_rollout(self):
        history = frame_for(200)
        future = frame_for(337, with_target=False, start="2023-01-09 08:00:00")
        preprocessing = fit_preprocessing(history)
        config = {
            "model": "dlinear",
            "context_length": 168,
            "prediction_length": 336,
            "moving_average": 25,
            "num_features": transformed_feature_count(preprocessing),
            "num_series": 1,
            "use_series_embedding": False,
            "prediction_floor": 0.0,
        }
        checkpoint = {
            "config": config,
            "preprocessing": preprocessing,
            "series_mapping": {"unit_000": 0},
            "state_dict": build_model(config).state_dict(),
        }
        with self.assertRaisesRegex(ValueError, "forbids autoregressive target rollout"):
            forecast(
                history, future[["series_id", "timestamp"]], checkpoint, future
            )


if __name__ == "__main__":
    unittest.main()
