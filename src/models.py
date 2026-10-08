"""Forecasting models behind one common interface.

Every forecaster implements:
    fit(df)              - df has 'ds' (datetime) and 'y' (float) columns
    predict(horizon)     - returns np.ndarray of length horizon
    interval(horizon)    - optional (lower, upper) arrays, None if unsupported
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .features import make_supervised


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


class HoltWintersForecaster(BaseForecaster):
    """Additive Holt-Winters exponential smoothing with prediction intervals.

    Intervals come from the in-sample residual variance and widen with
    sqrt(horizon), the standard approximation for ETS models.
    """

    name = "holt_winters"
    has_interval = True

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods

    def fit(self, df: pd.DataFrame) -> "HoltWintersForecaster":
        super().fit(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = ExponentialSmoothing(
                self._y,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=True)
        self._sigma = float(np.std(self._res.resid))
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._res.forecast(horizon))

    def interval(self, horizon: int, alpha: float = 0.05):
        from statistics import NormalDist

        z = NormalDist().inv_cdf(1 - alpha / 2)
        preds = self.predict(horizon)
        half_width = z * self._sigma * np.sqrt(np.arange(1, horizon + 1))
        return preds - half_width, preds + half_width


class ARIMAForecaster(BaseForecaster):
    """ARIMA(p,d,q) with prediction intervals from the state-space model."""

    name = "arima"
    has_interval = True

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)):
        self.order = order

    def fit(self, df: pd.DataFrame) -> "ARIMAForecaster":
        super().fit(df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = ARIMA(self._y, order=self.order).fit()
        return self

    def _get_forecast(self, horizon: int):
        return self._res.get_forecast(steps=horizon)

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self._get_forecast(horizon).predicted_mean)

    def interval(self, horizon: int, alpha: float = 0.05):
        ci = self._get_forecast(horizon).conf_int(alpha=alpha).to_numpy()
        return ci[:, 0], ci[:, 1]


class LinearLagsForecaster(BaseForecaster):
    """Linear regression on lag/rolling/calendar features, applied recursively."""

    name = "linear_lags"

    def __init__(self, lags: tuple[int, ...] = (1, 7, 14, 28)):
        self.lags = tuple(lags)
        self._model = LinearRegression()

    def fit(self, df: pd.DataFrame) -> "LinearLagsForecaster":
        super().fit(df)
        frame = pd.DataFrame({"ds": pd.to_datetime(df["ds"]), "y": self._y})
        X, y = make_supervised(frame, lags=self.lags)
        self._feature_cols = list(X.columns)
        self._model.fit(X, y)
        return self

    def _feature_row(self, history: list[float], ds: pd.Timestamp) -> list[float]:
        row = []
        arr = np.asarray(history, dtype=float)
        for k in self.lags:
            row.append(arr[-k] if len(arr) >= k else arr[0])
            window = arr[-k:] if len(arr) >= k else arr
            row.append(float(window.mean()))
            row.append(float(window.std()) if len(window) > 1 else 0.0)
        dow = ds.dayofweek
        row += [
            float(dow),
            float(ds.month),
            float(1 if dow >= 5 else 0),
            float(np.sin(2 * np.pi * dow / 7.0)),
            float(np.cos(2 * np.pi * dow / 7.0)),
        ]
        return row

    def predict(self, horizon: int) -> np.ndarray:
        history = list(self._y.to_numpy())
        future_ds = self._future_ds(horizon)
        preds = []
        for i in range(horizon):
            row = self._feature_row(history, future_ds[i])
            X = pd.DataFrame([row], columns=self._feature_cols)
            pred = float(self._model.predict(X)[0])
            preds.append(pred)
            history.append(pred)
        return np.array(preds)


MODELS = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "holt_winters": HoltWintersForecaster,
    "arima": ARIMAForecaster,
    "linear_lags": LinearLagsForecaster,
}


def get_model(name: str, **params) -> BaseForecaster:
    """Instantiate a forecaster by registry name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list_models()}")
    return MODELS[name](**params)


def list_models() -> list[str]:
    return list(MODELS.keys())
