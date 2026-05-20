from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(y_true - y_pred) / denom) * 100)


def _coerce_numeric_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float, copy=False)
    y_pred_arr = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float, copy=False)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    return y_true_arr[mask], y_pred_arr[mask]


def compute_metrics(y_true, y_pred) -> dict:
    y_true_arr, y_pred_arr = _coerce_numeric_pair(y_true, y_pred)
    if int(y_true_arr.size) == 0:
        nan = float("nan")
        return {"mae": nan, "rmse": nan, "mape": nan, "smape": nan}
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mape": mape(y_true_arr, y_pred_arr),
        "smape": smape(y_true_arr, y_pred_arr),
    }


def join_actual_forecast(
    df: pd.DataFrame, fcst: pd.DataFrame, id_col: str, time_col: str, target_col: str, model_col: str
) -> pd.DataFrame:
    out = df[[id_col, time_col, target_col]].merge(
        fcst[[id_col, time_col, model_col]], on=[id_col, time_col], how="inner"
    )
    return out
