import numpy as np
import pytest

from src.metrics import all_metrics, mae, mape, rmse, smape


def test_perfect_predictions_are_zero():
    y = [1.0, 2.0, 3.0, 4.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_known_values():
    assert mae([0, 0], [1, 3]) == 2.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert smape([100], [110]) == pytest.approx(200 * 10 / 210, rel=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2], [1, 2, 3])


def test_all_metrics_keys():
    m = all_metrics([1, 2, 3], [1, 2, 4])
    assert set(m) == {"mae", "rmse", "mape", "smape"}
    assert m["rmse"] >= m["mae"]
