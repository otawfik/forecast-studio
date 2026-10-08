"""Forecast Studio demo: pick a dataset and model, get a forecast with intervals."""
from __future__ import annotations

import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request

from src.data import PRESETS, generate_series, load_csv
from src.forecast import ForecastStudio
from src.models import list_models

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {name: f"Synthetic: {name}" for name in PRESETS}
DATASETS["sample"] = "Sample CSV (data/sample.csv)"


def load_dataset(name: str) -> pd.DataFrame:
    if name in PRESETS:
        return generate_series(**PRESETS[name])
    if name == "sample":
        path = os.path.join(BASE_DIR, "data", "sample.csv")
        return load_csv(path)
    raise ValueError(f"Unknown dataset '{name}'")


def forecast_chart(history: pd.DataFrame, fcst: pd.DataFrame, model_name: str) -> str:
    """Render history + forecast + interval band, return base64 PNG."""
    hist = history.tail(120)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(hist["ds"], hist["y"], color="#94a3b8", lw=1.5, label="history")
    ax.plot(fcst["ds"], fcst["yhat"], color="#2563eb", lw=2, label=f"forecast ({model_name})")
    if fcst["yhat_lower"].notna().any():
        ax.fill_between(
            fcst["ds"], fcst["yhat_lower"], fcst["yhat_upper"],
            color="#2563eb", alpha=0.18, label="95% interval",
        )
    ax.set_title(f"Forecast Studio: {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
def index():
    return render_template(
        "index.html",
        datasets=DATASETS,
        models=list_models(),
        result=None,
    )


@app.post("/forecast")
def forecast():
    dataset = request.form.get("dataset", "retail")
    model_name = request.form.get("model", "holt_winters")
    try:
        horizon = max(7, min(90, int(request.form.get("horizon", 30))))
    except (TypeError, ValueError):
        horizon = 30
    try:
        df = load_dataset(dataset)
        studio = ForecastStudio(model=model_name).fit(df)
        fcst = studio.predict(horizon=horizon)
        bt = studio.backtest(horizon=min(14, horizon), n_folds=3)
        chart = forecast_chart(df, fcst, model_name)
        result = {
            "dataset": dataset,
            "model": model_name,
            "horizon": horizon,
            "n_points": len(df),
            "chart": chart,
            "metrics": {
                "MAE": round(float(bt["mae"].mean()), 2),
                "RMSE": round(float(bt["rmse"].mean()), 2),
                "MAPE": round(float(bt["mape"].mean()), 2),
                "sMAPE": round(float(bt["smape"].mean()), 2),
            },
            "table": fcst.head(10).copy(),
        }
        result["table"]["ds"] = result["table"]["ds"].dt.strftime("%Y-%m-%d")
        for c in ("yhat", "yhat_lower", "yhat_upper"):
            result["table"][c] = result["table"][c].round(2)
        error = None
    except Exception as exc:  # noqa: BLE001 - demo should never 500
        result, error = None, str(exc)
    return render_template(
        "index.html", datasets=DATASETS, models=list_models(),
        result=result, error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
