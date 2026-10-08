import pandas as pd

from src.data import generate_series
from src.features import add_calendar_features, add_lag_features, make_supervised


def test_add_lag_features_columns():
    df = generate_series(n=60, seed=1)
    out = add_lag_features(df, lags=(1, 7))
    for col in ("lag_1", "lag_7", "roll_mean_1", "roll_mean_7", "roll_std_1", "roll_std_7"):
        assert col in out.columns
    # first lag_7 value should equal y[0]
    assert out["lag_7"].iloc[7] == df["y"].iloc[0]


def test_add_calendar_features():
    df = generate_series(n=30, seed=1, start="2025-01-06")  # a Monday
    out = add_calendar_features(df)
    assert out["dow"].iloc[0] == 0
    assert out["is_weekend"].iloc[5] == 1
    assert {"dow_sin", "dow_cos", "month"}.issubset(out.columns)


def test_make_supervised_drops_warmup():
    df = generate_series(n=100, seed=1)
    X, y = make_supervised(df, lags=(1, 7))
    assert len(X) == len(y)
    assert len(X) < 100  # warmup rows dropped
    assert X.notna().all().all()
    assert "lag_7" in X.columns and "dow" in X.columns
