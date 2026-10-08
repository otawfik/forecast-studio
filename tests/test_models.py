import numpy as np
import pytest

from src.data import generate_series
from src.models import get_model, list_models

DF = generate_series(n=120, seed=11)


def test_registry():
    assert set(list_models()) == {
        "naive", "seasonal_naive", "moving_average",
        "holt_winters", "arima", "linear_lags",
    }
    with pytest.raises(ValueError):
        get_model("nope")


@pytest.mark.parametrize("name", list_models())
def test_predict_length(name):
    model = get_model(name).fit(DF)
    preds = model.predict(14)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 14
    assert np.isfinite(preds).all()


def test_naive_repeats_last():
    preds = get_model("naive").fit(DF).predict(5)
    assert (preds == DF["y"].iloc[-1]).all()


def test_seasonal_naive_repeats_cycle():
    preds = get_model("seasonal_naive", seasonal_period=7).fit(DF).predict(9)
    expected = DF["y"].iloc[-7:].to_numpy()
    np.testing.assert_allclose(preds[:7], expected)
    np.testing.assert_allclose(preds[7:], expected[:2])


def test_moving_average():
    preds = get_model("moving_average", window=7).fit(DF).predict(3)
    assert (preds == DF["y"].iloc[-7:].mean()).all()


def test_intervals_sane():
    for name in ("holt_winters", "arima"):
        model = get_model(name).fit(DF)
        iv = model.interval(10)
        assert iv is not None
        lo, hi = iv
        assert len(lo) == 10 and len(hi) == 10
        assert (lo <= hi).all()
        assert np.isfinite(lo).all() and np.isfinite(hi).all()


def test_baselines_have_no_interval():
    for name in ("naive", "seasonal_naive", "moving_average", "linear_lags"):
        assert get_model(name).fit(DF).interval(5) is None


def test_too_short_raises():
    with pytest.raises(ValueError):
        get_model("naive").fit(DF.head(5))
