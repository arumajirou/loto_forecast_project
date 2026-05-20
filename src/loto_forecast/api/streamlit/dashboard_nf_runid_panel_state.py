from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def resolve_selected_model(
    *,
    model_df: pd.DataFrame,
    sel_run: str,
    parse_json_like: Any,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    sel_model = pd.DataFrame()
    if not model_df.empty and "run_id" in model_df.columns:
        sel_model = model_df[model_df["run_id"].astype(str) == str(sel_run)].copy()
    sel_model_row = sel_model.iloc[0].to_dict() if not sel_model.empty else {}
    sel_params = parse_json_like(sel_model_row.get("params_json")) if sel_model_row else {}
    sel_metrics = parse_json_like(sel_model_row.get("metrics_json")) if sel_model_row else {}
    if not isinstance(sel_params, dict):
        sel_params = {}
    if not isinstance(sel_metrics, dict):
        sel_metrics = {}
    return sel_model, sel_model_row, sel_params, sel_metrics


def build_file_rows(sel_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if sel_dir.exists() and sel_dir.is_dir():
        for path in sorted(sel_dir.rglob("*")):
            if path.is_file():
                rows.append({"file": str(path.relative_to(sel_dir)), "size_bytes": int(path.stat().st_size)})
    return pd.DataFrame(rows)


def config_context(sel_meta: dict[str, Any], sel_model_row: dict[str, Any], sel_params: dict[str, Any]) -> dict[str, Any]:
    meta_params = sel_meta.get("model_params", {}) if isinstance(sel_meta.get("model_params"), dict) else {}
    meta_runtime = sel_meta.get("nf_runtime_kwargs", {}) if isinstance(sel_meta.get("nf_runtime_kwargs"), dict) else {}
    return {
        "meta_model_name": sel_meta.get("model_name"),
        "db_model_name": sel_model_row.get("model_name"),
        "meta_h": sel_meta.get("h"),
        "db_h": sel_model_row.get("horizon"),
        "expected_backend": meta_params.get("backend", sel_meta.get("backend")),
        "expected_num_samples": meta_params.get("num_samples", sel_meta.get("num_samples")),
        "actual_backend": sel_params.get("backend"),
        "actual_num_samples": sel_params.get("num_samples"),
        "meta_pred_h": meta_runtime.get("nf_predict_kwargs", {}).get("h")
        if isinstance(meta_runtime.get("nf_predict_kwargs"), dict)
        else None,
        "meta_cv_h": meta_runtime.get("nf_cross_validation_kwargs", {}).get("h")
        if isinstance(meta_runtime.get("nf_cross_validation_kwargs"), dict)
        else None,
    }


def mismatch_flags(*, meta_h: Any, meta_pred_h: Any, meta_cv_h: Any, safe_int_eq: Any) -> tuple[bool, bool]:
    return safe_int_eq(meta_h, meta_pred_h) is False, safe_int_eq(meta_h, meta_cv_h) is False


def default_metric_name(metric_df: pd.DataFrame) -> tuple[list[str], str]:
    if metric_df.empty:
        return [], ""
    options = sorted(metric_df["metric"].astype(str).unique().tolist())
    focus_metrics = ["mae", "rmse", "mape", "smape"]
    default_metric = next((metric for metric in focus_metrics if metric in options), options[0])
    return options, default_metric


def default_target_metric(metric_cols: list[str]) -> str | None:
    if not metric_cols:
        return None
    pref = ["metric.mae", "metric.rmse", "metric.mape"]
    return next((col for col in pref if col in metric_cols), metric_cols[0])


def treatment_candidates(corr_df: pd.DataFrame, *, limit: int = 20) -> list[str]:
    if corr_df.empty or "feature" not in corr_df.columns:
        return []
    return corr_df["feature"].head(limit).astype(str).tolist()
