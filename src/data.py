"""Data loading and synthetic series generation.

Every series is a DataFrame with two columns:
    ds : datetime64  - the timestamp
    y  : float       - the observed value
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRESETS = {
    "retail": dict(
        n=365, base=120.0, trend=0.08, seasonal_period=7,
        seasonal_amp=18.0, noise=6.0, seed=7, yearly=False,
    ),
    "traffic": dict(
        n=365, base=2500.0, trend=0.15, seasonal_period=7,
        seasonal_amp=400.0, noise=180.0, seed=21, yearly=False,
    ),
    "energy": dict(
        n=365, base=300.0, trend=0.02, seasonal_period=7,
        seasonal_amp=30.0, noise=12.0, seed=99, yearly=True,
    ),
}


def generate_series(
    n: int = 365,
    base: float = 100.0,
    trend: float = 0.05,
    seasonal_period: int = 7,
    seasonal_amp: float = 10.0,
    noise: float = 2.0,
    yearly: bool = False,
    seed: int = 42,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    """Generate a synthetic series: trend + weekly seasonality + noise.

    Set yearly=True to add an annual seasonal component on top.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = base + trend * t
    y += seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
    if yearly:
        y += 0.6 * seasonal_amp * np.sin(2 * np.pi * t / 365.25 + 1.0)
    y += rng.normal(0.0, noise, size=n)
    ds = pd.date_range(start=start, periods=n, freq="D")
    return pd.DataFrame({"ds": ds, "y": y.astype(float)})


def load_csv(path: str, date_col: str = "ds", value_col: str = "y") -> pd.DataFrame:
    """Load a CSV with a date column and a value column into ds/y format."""
    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{date_col}' and '{value_col}' columns, "
            f"got {list(df.columns)}"
        )
    out = pd.DataFrame({
        "ds": pd.to_datetime(df[date_col]),
        "y": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna().sort_values("ds").reset_index(drop=True)
    if len(out) < 30:
        raise ValueError(f"Need at least 30 observations, got {len(out)}")
    return out


def train_test_split(df: pd.DataFrame, test_size: int = 30):
    """Split into train (all but last test_size rows) and test."""
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and len(df) - 1")
    train = df.iloc[:-test_size].reset_index(drop=True)
    test = df.iloc[-test_size:].reset_index(drop=True)
    return train, test
