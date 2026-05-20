from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config.settings import settings


def plot_forecast_vs_actual(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_col: str,
    output_path: str | Path,
    id_value: str | None = None,
) -> Path:
    id_col = settings.id_col
    time_col = settings.time_col
    target_col = settings.target_col

    act = actual_df.copy()
    fc = forecast_df.copy()

    if id_value is not None and id_col in act.columns and id_col in fc.columns:
        act = act[act[id_col] == id_value]
        fc = fc[fc[id_col] == id_value]

    merged = act[[id_col, time_col, target_col]].merge(
        fc[[id_col, time_col, model_col]], on=[id_col, time_col], how="inner"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged[time_col], merged[target_col], label="actual", linewidth=2)
    ax.plot(merged[time_col], merged[model_col], label=model_col, linewidth=2)
    ax.set_title("Actual vs Forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_exog_importance(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    value_col: str = "delta_mae_mean",
    feature_col: str = "feature",
    top_n: int = 20,
) -> Path:
    if importance_df.empty:
        raise ValueError("importance_df is empty")

    top = importance_df.sort_values(value_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(top[feature_col].astype(str), top[value_col].astype(float))
    ax.invert_yaxis()
    ax.set_title("Exogenous Feature Importance")
    ax.set_xlabel(value_col)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
