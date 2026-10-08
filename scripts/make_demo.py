"""Generate demo assets: data/sample.csv and screenshots/demo.svg.

The SVG is a lightweight hand-built mockup of the dashboard, drawn from a real
Holt-Winters forecast so the curve shapes are genuine.
Run from the repo root:  python scripts/make_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import PRESETS, generate_series
from src.forecast import ForecastStudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_sample_csv():
    df = generate_series(**PRESETS["retail"])
    path = os.path.join(ROOT, "data", "sample.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


def _polyline(xs, ys, x0, x1, y0, y1, w, h):
    pts = []
    for x, y in zip(xs, ys):
        px = 40 + (x - x0) / (x1 - x0) * (w - 80)
        py = h - 50 - (y - y0) / (y1 - y0) * (h - 110)
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def write_demo_svg():
    df = generate_series(**PRESETS["retail"])
    studio = ForecastStudio(model="holt_winters").fit(df)
    fcst = studio.predict(horizon=30)
    bt = studio.backtest(horizon=14, n_folds=3)
    rmse = float(bt["rmse"].mean())
    mape = float(bt["mape"].mean())

    hist = df.tail(90)
    W, H = 900, 420
    all_y = np.concatenate([hist["y"].to_numpy(), fcst["yhat"].to_numpy()])
    y0, y1 = float(all_y.min()) * 0.92, float(all_y.max()) * 1.08
    n_hist = len(hist)

    hist_line = _polyline(range(n_hist), hist["y"].to_numpy(), 0, n_hist + 30, y0, y1, W, H)
    fcst_line = _polyline(
        range(n_hist - 1, n_hist + 30), np.concatenate([[hist["y"].iloc[-1]], fcst["yhat"].to_numpy()]),
        0, n_hist + 30, y0, y1, W, H,
    )
    band_top = _polyline(
        range(n_hist - 1, n_hist + 30), np.concatenate([[hist["y"].iloc[-1]], fcst["yhat_upper"].to_numpy()]),
        0, n_hist + 30, y0, y1, W, H,
    )
    band_bot = _polyline(
        range(n_hist - 1, n_hist + 30), np.concatenate([[hist["y"].iloc[-1]], fcst["yhat_lower"].to_numpy()]),
        0, n_hist + 30, y0, y1, W, H,
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" rx="12" fill="#0f172a"/>
<text x="28" y="38" font-family="system-ui,sans-serif" font-size="22" font-weight="bold" fill="#e2e8f0">Forecast Studio</text>
<text x="28" y="62" font-family="system-ui,sans-serif" font-size="13" fill="#94a3b8">holt_winters on retail &#183; 30-day forecast with 95% interval</text>
<g font-family="system-ui,sans-serif">
<rect x="640" y="18" width="110" height="56" rx="8" fill="#1e293b"/>
<text x="652" y="40" font-size="11" fill="#94a3b8">RMSE</text>
<text x="652" y="62" font-size="20" font-weight="bold" fill="#e2e8f0">{rmse:.1f}</text>
<rect x="760" y="18" width="110" height="56" rx="8" fill="#1e293b"/>
<text x="772" y="40" font-size="11" fill="#94a3b8">MAPE</text>
<text x="772" y="62" font-size="20" font-weight="bold" fill="#e2e8f0">{mape:.1f}%</text>
</g>
<polygon points="{band_top} {band_bot}" fill="#2563eb" opacity="0.18"/>
<polyline points="{hist_line}" fill="none" stroke="#94a3b8" stroke-width="2"/>
<polyline points="{fcst_line}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<text x="40" y="{H - 18}" font-family="system-ui,sans-serif" font-size="12" fill="#94a3b8">history</text>
<circle cx="105" cy="{H - 22}" r="5" fill="#94a3b8"/>
<text x="120" y="{H - 18}" font-family="system-ui,sans-serif" font-size="12" fill="#94a3b8">forecast</text>
<circle cx="190" cy="{H - 22}" r="5" fill="#2563eb"/>
</svg>"""
    path = os.path.join(ROOT, "screenshots", "demo.svg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(svg)
    print(f"wrote {path} ({len(svg)} bytes)")


if __name__ == "__main__":
    write_sample_csv()
    write_demo_svg()
