import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Forecast Studio" in resp.data


def test_forecast_post(client):
    resp = client.post(
        "/forecast",
        data={"dataset": "retail", "model": "seasonal_naive", "horizon": "14"},
    )
    assert resp.status_code == 200
    assert b"data:image/png;base64" in resp.data
    assert b"RMSE" in resp.data
