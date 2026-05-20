from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config.settings import settings
from ..data.db import make_engine, read_timeseries
from ..features.engineering import add_cyclical_time_features, add_time_features, infer_exog_columns, make_future_df
from ..infra.meta_store import write_exog_contribution
from .diagnostics import granger_test
from .evaluation import compute_metrics


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {settings.id_col, settings.time_col, settings.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = df.copy()
    out[settings.time_col] = pd.to_datetime(out[settings.time_col], errors="coerce")
    out = out.dropna(subset=[settings.id_col, settings.time_col, settings.target_col])
    out = out.drop_duplicates(subset=[settings.id_col, settings.time_col], keep="last")
    out = out.sort_values([settings.id_col, settings.time_col]).reset_index(drop=True)
    if out.empty:
        raise ValueError("dataset is empty after required-column filtering")
    return out


def permutation_importance_exog(
    run_id: str,
    h: int | None = None,
    n_repeats: int = 3,
    dataset_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Permutation-based exogenous importance as a robust fallback."""
    from ..models.neuralforecast_model import load_model, predict_with_model, prepare_nf_frames

    run_dir = settings.artifact_dir / run_id
    nf = load_model(run_dir)

    engine = make_engine()
    raw = (
        dataset_df.copy() if dataset_df is not None else read_timeseries(engine, settings.db_schema, settings.db_table)
    )
    df = _prepare(raw)

    meta = json_load(run_dir / "meta.json")
    h = h or int(meta.get("h", settings.default_horizon))
    exog = meta.get("exog") if isinstance(meta.get("exog"), dict) else infer_exog_columns(df)

    test = df.groupby(settings.id_col).tail(h)
    train_df = df.drop(test.index)
    if train_df.empty:
        raise ValueError("not enough rows for permutation importance: train split is empty")

    futr = make_future_df(train_df, h=h, freq=settings.freq)
    futr = add_time_features(futr)
    futr = add_cyclical_time_features(futr)
    train_fit, futr_fit, exog_fit = prepare_nf_frames(df=train_df, exog=exog, futr_df=futr)

    base_fcst = predict_with_model(nf, df=train_fit, futr_df=futr_fit)
    model_col = [c for c in base_fcst.columns if c not in (settings.id_col, settings.time_col)][0]
    merged = test[[settings.id_col, settings.time_col, settings.target_col]].merge(
        base_fcst[[settings.id_col, settings.time_col, model_col]],
        on=[settings.id_col, settings.time_col],
        how="inner",
    )
    if merged.empty:
        raise ValueError("no overlap between forecast and test windows in permutation importance")
    base = compute_metrics(merged[settings.target_col], merged[model_col])

    candidates = list(dict.fromkeys(exog_fit.get("hist_exog", []) + exog_fit.get("futr_exog", [])))
    rows = []
    for col in candidates:
        if col not in train_fit.columns and (futr_fit is None or col not in futr_fit.columns):
            continue
        deltas = []
        for _ in range(n_repeats):
            train_shuf = train_fit.copy()
            futr_shuf = futr_fit.copy() if futr_fit is not None else None
            if col in train_shuf.columns:
                train_shuf[col] = np.random.permutation(train_shuf[col].values)
            if futr_shuf is not None and col in futr_shuf.columns:
                futr_shuf[col] = np.random.permutation(futr_shuf[col].values)

            fc = predict_with_model(nf, df=train_shuf, futr_df=futr_shuf)
            merged2 = test[[settings.id_col, settings.time_col, settings.target_col]].merge(
                fc[[settings.id_col, settings.time_col, model_col]],
                on=[settings.id_col, settings.time_col],
                how="inner",
            )
            if merged2.empty:
                continue
            m = compute_metrics(merged2[settings.target_col], merged2[model_col])
            deltas.append(m["mae"] - base["mae"])

        if not deltas:
            continue
        rows.append(
            {
                "feature": col,
                "delta_mae_mean": float(np.mean(deltas)),
                "delta_mae_std": float(np.std(deltas)),
                "base_mae": float(base["mae"]),
                "method": "permutation",
            }
        )

    if rows:
        out = pd.DataFrame(rows).sort_values("delta_mae_mean", ascending=False)
    else:
        out = pd.DataFrame(columns=["feature", "delta_mae_mean", "delta_mae_std", "base_mae", "method"])
    out_path = run_dir / "exog_importance.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(f"exog importance saved: {out_path}")

    try:
        write_exog_contribution(engine, run_id, out)
    except Exception as e:
        logger.warning(f"write_exog_contribution failed: {e}")

    return out


def exog_granger_screening(maxlag: int = 8, top_k: int = 20) -> pd.DataFrame:
    """Statistical screening of exogenous usefulness by Granger p-values."""
    engine = make_engine()
    raw = read_timeseries(engine, settings.db_schema, settings.db_table)
    df = _prepare(raw)

    candidates = [
        c
        for c in df.columns
        if c not in {settings.id_col, settings.time_col, settings.target_col} and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    if not candidates:
        return pd.DataFrame(columns=["feature", "min_pvalue", "best_lag"])

    # For multi-series, concatenate as pragmatic approximation.
    single = df.sort_values([settings.id_col, settings.time_col]).copy()
    for col in candidates:
        pvals = granger_test(single, y_col=settings.target_col, x_col=col, maxlag=maxlag)
        if not pvals:
            continue
        lag, p = min(((int(k), float(v)) for k, v in pvals.items()), key=lambda x: x[1])
        rows.append({"feature": col, "min_pvalue": p, "best_lag": lag})

    out = pd.DataFrame(rows).sort_values(["min_pvalue", "feature"]).head(top_k)
    return out


def neuralforecast_explainability(run_id: str) -> str:
    """Try NeuralForecast explainability API if available in current version."""
    _ = settings.artifact_dir / run_id
    try:
        from neuralforecast.explainability import Explainability  # noqa: F401
    except Exception as e:
        return f"NeuralForecast explainability import failed: {e}"

    return "Explainability is available. Use notebooks/02_explainability_and_tests.ipynb for runnable examples."


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
