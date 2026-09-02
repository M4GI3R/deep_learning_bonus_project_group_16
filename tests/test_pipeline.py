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
from data import FEATURE_COLUMNS, WindowDataset, fit_preprocessing, transform_features
from predict import forecast
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
        self.assertEqual(transformed.shape, (200, len(FEATURE_COLUMNS)))
        self.assertTrue(np.isfinite(transformed).all())

    def test_multivariate_window_and_models(self):
        frame = frame_for(200)
        preprocessing = fit_preprocessing(frame)
        dataset = WindowDataset(frame, 168, 24, 24, preprocessing, {"unit_000": 0})
        x, _y, series, history_features, future_features = dataset[0]
        common = {
            "context_length": 168,
            "prediction_length": 24,
            "moving_average": 25,
            "num_features": len(FEATURE_COLUMNS),
            "num_series": 1,
        }
        dlinear = build_model(
            {**common, "model": "dlinear", "use_series_embedding": False}
        )
        output = dlinear(
            x[None],
            history_features=history_features[None],
            future_features=future_features[None],
        )
        self.assertEqual(output.shape, (1, 24))
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
        self.assertEqual(output.shape, (1, 24))

    def test_forecast_covers_full_multiblock_index(self):
        history = frame_for(200)
        future = frame_for(48, with_target=False, start="2023-01-09 08:00:00")
        preprocessing = fit_preprocessing(history)
        config = {
            "model": "dlinear",
            "context_length": 168,
            "prediction_length": 24,
            "moving_average": 25,
            "num_features": len(FEATURE_COLUMNS),
            "num_series": 1,
            "use_series_embedding": False,
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
        self.assertEqual(len(result), 48)
        self.assertTrue(np.isfinite(result["prediction"]).all())


if __name__ == "__main__":
    unittest.main()
