"""Forecasting models behind one common interface.

Every forecaster implements:
    fit(df)              - df has 'ds' (datetime) and 'y' (float) columns
    predict(horizon)     - returns np.ndarray of length horizon
    interval(horizon)    - optional (lower, upper) arrays, None if unsupported
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BaseForecaster:
    name = "base"
    has_interval = False

    def fit(self, df: pd.DataFrame) -> "BaseForecaster":
        df = df.copy()
        self._y = pd.Series(df["y"]).reset_index(drop=True).astype(float)
        self._last_ds = pd.to_datetime(df["ds"]).iloc[-1]
        if len(self._y) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(self._y)}")
        return self

    def _future_ds(self, horizon: int) -> pd.DatetimeIndex:
        return pd.date_range(
            self._last_ds + pd.Timedelta(days=1), periods=horizon, freq="D"
        )

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def interval(self, horizon: int, alpha: float = 0.05):
        """(lower, upper) prediction intervals, or None if unsupported."""
        return None


class NaiveForecaster(BaseForecaster):
    """Repeat the last observed value."""

    name = "naive"

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._y.iloc[-1])


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the last seasonal cycle (e.g. same weekday last week)."""

    name = "seasonal_naive"

    def __init__(self, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period

    def predict(self, horizon: int) -> np.ndarray:
        cycle = self._y.iloc[-self.seasonal_period :].to_numpy()
        reps = int(np.ceil(horizon / self.seasonal_period))
        return np.tile(cycle, reps)[:horizon]


class MovingAverageForecaster(BaseForecaster):
    """Repeat the mean of the last `window` observations."""

    name = "moving_average"

    def __init__(self, window: int = 7):
        self.window = window

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._y.iloc[-self.window :].mean())


MODELS = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
}


def get_model(name: str, **params) -> BaseForecaster:
    """Instantiate a forecaster by registry name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list_models()}")
    return MODELS[name](**params)


def list_models() -> list[str]:
    return list(MODELS.keys())
