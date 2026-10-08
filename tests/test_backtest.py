import pandas as pd
import pytest

from src.backtest import compare_models, walk_forward
from src.data import generate_series

DF = generate_series(n=200, seed=13)


def test_walk_forward_folds():
    folds = walk_forward(DF, "seasonal_naive", horizon=14, n_folds=4)
    assert len(folds) == 4
    assert set(folds.columns) >= {"fold", "mae", "rmse", "mape", "smape"}
    assert (folds["rmse"] >= 0).all()


def test_walk_forward_not_enough_data():
    with pytest.raises(ValueError):
        walk_forward(DF.head(40), "naive", horizon=14, n_folds=5)


def test_compare_models_sorted_by_rmse():
    table = compare_models(
        DF, model_names=["naive", "seasonal_naive", "moving_average"],
        horizon=14, n_folds=3,
    )
    assert list(table.columns) == ["model", "mae", "rmse", "mape", "smape"]
    assert (table["rmse"].diff().dropna() >= 0).all()
