"""End-to-end forecasting pipeline: fit once, forecast, backtest, compare."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import compare_models, walk_forward
from .models import get_model, list_models


class ForecastStudio:
    """High-level API over the model registry.

    Usage:
        studio = ForecastStudio(model="holt_winters").fit(df)
        fcst = studio.predict(horizon=30)      # ds, yhat, yhat_lower, yhat_upper
        print(studio.backtest(horizon=14))    # walk-forward metrics per fold
        print(studio.compare())               # every model, sorted by RMSE
    """

    def __init__(self, model: str = "holt_winters", **params):
        if model not in list_models():
            raise ValueError(f"Unknown model '{model}'. Choose from {list_models()}")
        self.model_name = model
        self.params = params
        self._model = None
        self._df = None

    def fit(self, df: pd.DataFrame) -> "ForecastStudio":
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        self._df = df.sort_values("ds").reset_index(drop=True)
        self._model = get_model(self.model_name, **self.params).fit(self._df)
        return self

    def predict(self, horizon: int = 30) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        preds = self._model.predict(horizon)
        future_ds = pd.date_range(
            self._df["ds"].iloc[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        out = pd.DataFrame({"ds": future_ds, "yhat": np.asarray(preds, dtype=float)})
        interval = self._model.interval(horizon)
        if interval is not None:
            lo, hi = interval
            out["yhat_lower"] = np.asarray(lo, dtype=float)
            out["yhat_upper"] = np.asarray(hi, dtype=float)
        else:
            out["yhat_lower"] = np.nan
            out["yhat_upper"] = np.nan
        return out

    def backtest(self, horizon: int = 14, n_folds: int = 5) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call fit() before backtest()")
        return walk_forward(
            self._df, self.model_name, horizon=horizon, n_folds=n_folds, **self.params
        )

    def compare(
        self, models: list[str] | None = None, horizon: int = 14, n_folds: int = 5
    ) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call fit() before compare()")
        return compare_models(self._df, models, horizon=horizon, n_folds=n_folds)
