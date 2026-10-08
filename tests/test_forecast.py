import pandas as pd
import pytest

from src.data import generate_series
from src.forecast import ForecastStudio

DF = generate_series(n=200, seed=17)


def test_predict_frame():
    fcst = ForecastStudio(model="holt_winters").fit(DF).predict(horizon=21)
    assert list(fcst.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(fcst) == 21
    assert fcst["ds"].iloc[0] == DF["ds"].iloc[-1] + pd.Timedelta(days=1)
    assert (fcst["yhat_lower"] <= fcst["yhat_upper"]).all()


def test_predict_without_interval_model():
    fcst = ForecastStudio(model="naive").fit(DF).predict(horizon=7)
    assert fcst["yhat_lower"].isna().all()


def test_backtest_runs():
    bt = ForecastStudio(model="seasonal_naive").fit(DF).backtest(horizon=14, n_folds=3)
    assert len(bt) == 3


def test_compare_runs():
    table = ForecastStudio(model="naive").fit(DF).compare(
        models=["naive", "moving_average"], horizon=14, n_folds=2
    )
    assert set(table["model"]) == {"naive", "moving_average"}


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        ForecastStudio(model="prophet")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ForecastStudio().predict(7)
