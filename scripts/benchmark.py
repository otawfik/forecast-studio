"""Benchmark every model on every synthetic preset with walk-forward backtesting.

Writes models/benchmark.json and prints a markdown table for the README.
Run from the repo root:  python scripts/benchmark.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import compare_models
from src.data import PRESETS, generate_series

HORIZON = 14
N_FOLDS = 5


def to_markdown(df: "pd.DataFrame") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    results = {}
    for preset, kwargs in PRESETS.items():
        df = generate_series(**kwargs)
        table = compare_models(df, horizon=HORIZON, n_folds=N_FOLDS)
        results[preset] = table.round(2).to_dict(orient="records")
        print(f"## {preset} (horizon={HORIZON}, folds={N_FOLDS})")
        print(to_markdown(table.round(2)))
        print()
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "benchmark.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"horizon": HORIZON, "n_folds": N_FOLDS, "results": results}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
