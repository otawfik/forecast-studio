"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    return yt, yp


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (percent, zeros in y_true ignored)."""
    yt, yp = _as_arrays(y_true, y_pred)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error (percent)."""
    yt, yp = _as_arrays(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100.0)


def all_metrics(y_true, y_pred) -> dict:
    """All metrics as a dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
