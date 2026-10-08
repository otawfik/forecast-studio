import os
import tempfile

import pandas as pd
import pytest

from src.data import PRESETS, generate_series, load_csv, train_test_split


def test_generate_series_shape_and_columns():
    df = generate_series(n=120, seed=1)
    assert list(df.columns) == ["ds", "y"]
    assert len(df) == 120
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert df["y"].notna().all()


def test_generate_series_reproducible():
    a = generate_series(n=50, seed=3)
    b = generate_series(n=50, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_presets_load():
    for name, kwargs in PRESETS.items():
        df = generate_series(**kwargs)
        assert len(df) == kwargs["n"], name


def test_load_csv_roundtrip():
    df = generate_series(n=60, seed=5)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        df.to_csv(path, index=False)
        loaded = load_csv(path)
        assert list(loaded.columns) == ["ds", "y"]
        assert len(loaded) == 60
    finally:
        os.unlink(path)


def test_load_csv_bad_columns():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("a,b\n1,2\n")
        path = f.name
    try:
        with pytest.raises(ValueError):
            load_csv(path)
    finally:
        os.unlink(path)


def test_train_test_split():
    df = generate_series(n=100, seed=2)
    train, test = train_test_split(df, test_size=20)
    assert len(train) == 80 and len(test) == 20
    assert train["ds"].max() < test["ds"].min()
