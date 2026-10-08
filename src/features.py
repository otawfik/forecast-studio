"""Feature engineering for time series: lags, rolling stats, calendar effects."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = "y",
    lags: tuple[int, ...] = (1, 7, 14, 28),
) -> pd.DataFrame:
    """Add lag_k, roll_mean_k and roll_std_k columns for each lag k."""
    out = df.copy()
    y = out[value_col]
    for k in lags:
        out[f"lag_{k}"] = y.shift(k)
        rolled = y.shift(1).rolling(window=k, min_periods=1)
        out[f"roll_mean_{k}"] = rolled.mean()
        out[f"roll_std_{k}"] = rolled.std().fillna(0.0)
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    """Add day-of-week, month, weekend flag and cyclic encodings."""
    out = df.copy()
    ds = pd.to_datetime(out[date_col])
    out["dow"] = ds.dt.dayofweek
    out["month"] = ds.dt.month
    out["is_weekend"] = (ds.dt.dayofweek >= 5).astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    return out


def make_supervised(
    df: pd.DataFrame,
    value_col: str = "y",
    lags: tuple[int, ...] = (1, 7, 14, 28),
) -> tuple[pd.DataFrame, pd.Series]:
    """Build an (X, y) supervised frame from lag + calendar features.

    Rows with NaN (warmup from shifting) are dropped.
    """
    feat = add_lag_features(df, value_col=value_col, lags=lags)
    feat = add_calendar_features(feat)
    feat = feat.dropna().reset_index(drop=True)
    feature_cols = [c for c in feat.columns if c not in ("ds", value_col)]
    return feat[feature_cols], feat[value_col]
