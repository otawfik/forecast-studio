"""Walk-forward (expanding window) backtesting for forecasters."""
from __future__ import annotations

import pandas as pd

from .metrics import all_metrics
from .models import get_model, list_models


def walk_forward(
    df: pd.DataFrame,
    model_name: str,
    horizon: int = 14,
    n_folds: int = 5,
    step: int | None = None,
    **params,
) -> pd.DataFrame:
    """Expanding-window walk-forward validation.

    Fold i trains on the first (initial + i * step) rows and scores the next
    `horizon` rows. Returns one row per fold with accuracy metrics.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    step = step or horizon
    initial = n - horizon - (n_folds - 1) * step
    if initial < 30:
        raise ValueError(
            f"Not enough data for {n_folds} folds of horizon {horizon} "
            f"(need >= {30 + horizon + (n_folds - 1) * step} rows, got {n})"
        )
    rows = []
    for fold in range(n_folds):
        train_end = initial + fold * step
        train = df.iloc[:train_end]
        test = df.iloc[train_end : train_end + horizon]
        if len(test) < horizon:
            break
        model = get_model(model_name, **params).fit(train)
        preds = model.predict(horizon)
        metrics = all_metrics(test["y"].to_numpy(), preds)
        rows.append({"fold": fold + 1, "train_end": train_end, **metrics})
    return pd.DataFrame(rows)


def compare_models(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    horizon: int = 14,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Mean walk-forward metrics per model, sorted by RMSE (best first)."""
    model_names = model_names or list_models()
    rows = []
    for name in model_names:
        folds = walk_forward(df, name, horizon=horizon, n_folds=n_folds)
        rows.append({
            "model": name,
            "mae": folds["mae"].mean(),
            "rmse": folds["rmse"].mean(),
            "mape": folds["mape"].mean(),
            "smape": folds["smape"].mean(),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
