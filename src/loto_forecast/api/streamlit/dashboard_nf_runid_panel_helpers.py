from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_run_snapshot(
    *,
    sel_run: str,
    sel_dir: Path,
    sel_meta: dict[str, Any],
    sel_model: pd.DataFrame,
    sel_model_row: dict[str, Any],
    settings: Any,
    has_model_artifacts: Any,
) -> dict[str, Any]:
    return {
        "run_id": str(sel_run),
        "artifact_dir": str(sel_dir),
        "meta_model_name": sel_meta.get("model_name"),
        "meta_h": sel_meta.get("h"),
        "meta_runtime_kwargs": sel_meta.get("nf_runtime_kwargs", {}),
        "db_model_row_exists": bool(not sel_model.empty),
        "db_model_name": sel_model_row.get("model_name"),
        "db_horizon": sel_model_row.get("horizon"),
        "default_h": int(settings.default_horizon),
        "artifact_exists": bool(sel_dir.exists()),
        "model_files": bool(has_model_artifacts(sel_dir)),
    }


def build_config_check_rows(
    *,
    meta_model_name: Any,
    db_model_name: Any,
    meta_h: Any,
    db_h: Any,
    expected_backend: Any,
    actual_backend: Any,
    expected_num_samples: Any,
    actual_num_samples: Any,
    meta_pred_h: Any,
    meta_cv_h: Any,
    safe_int_eq: Any,
) -> list[dict[str, Any]]:
    return [
        {
            "check": "model_name (meta vs DB)",
            "expected": meta_model_name,
            "actual": db_model_name,
            "ok": (str(meta_model_name) == str(db_model_name))
            if (meta_model_name is not None and db_model_name is not None)
            else None,
        },
        {
            "check": "h (meta.h vs model.horizon)",
            "expected": meta_h,
            "actual": db_h,
            "ok": safe_int_eq(meta_h, db_h),
        },
        {
            "check": "backend (meta.model_params vs model.params_json)",
            "expected": expected_backend,
            "actual": actual_backend if actual_backend is not None else "(DB params_json未記録)",
            "ok": (str(expected_backend) == str(actual_backend))
            if (expected_backend is not None and actual_backend is not None)
            else None,
        },
        {
            "check": "num_samples (meta.model_params vs model.params_json)",
            "expected": expected_num_samples,
            "actual": actual_num_samples if actual_num_samples is not None else "(DB params_json未記録)",
            "ok": safe_int_eq(expected_num_samples, actual_num_samples),
        },
        {
            "check": "predict h (meta.h vs meta.nf_predict_kwargs.h)",
            "expected": meta_h,
            "actual": meta_pred_h,
            "ok": safe_int_eq(meta_h, meta_pred_h),
        },
        {
            "check": "cross_validation h (meta.h vs meta.nf_cross_validation_kwargs.h)",
            "expected": meta_h,
            "actual": meta_cv_h,
            "ok": safe_int_eq(meta_h, meta_cv_h),
        },
    ]


def build_model_resource_df(model_resource_df: pd.DataFrame) -> pd.DataFrame:
    if model_resource_df.empty:
        return model_resource_df.copy()
    out = model_resource_df.copy()
    out["efficiency_score"] = (
        pd.to_numeric(out["avg_rows_written"], errors="coerce").fillna(0.0)
        / pd.to_numeric(out["avg_duration_sec"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    return out


def build_metric_rows(model_df: pd.DataFrame, *, parse_json_like: Any, row_limit: int) -> pd.DataFrame:
    metric_rows: list[dict[str, Any]] = []
    for row in model_df.head(max(3000, row_limit * 8)).to_dict(orient="records"):
        metrics_obj = parse_json_like(row.get("metrics_json"))
        if not isinstance(metrics_obj, dict):
            continue
        base_row = {
            "run_id": str(row.get("run_id", "")),
            "model_name": str(row.get("model_name", "")),
            "status": str(row.get("status", "")),
            "horizon": row.get("horizon"),
        }
        for key, value in metrics_obj.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metric_rows.append({**base_row, "metric": str(key), "value": float(value)})
    return pd.DataFrame(metric_rows)


def build_analysis_df(model_df: pd.DataFrame, *, parse_json_like: Any, row_limit: int) -> pd.DataFrame:
    analysis_rows: list[dict[str, Any]] = []
    for row in model_df.head(max(2000, row_limit * 5)).to_dict(orient="records"):
        rec: dict[str, Any] = {
            "run_id": str(row.get("run_id", "")),
            "model_name": str(row.get("model_name", "")),
            "status": str(row.get("status", "")),
        }
        if row.get("horizon") is not None:
            rec["horizon"] = pd.to_numeric(row.get("horizon"), errors="coerce")
        metrics_obj = parse_json_like(row.get("metrics_json"))
        params_obj = parse_json_like(row.get("params_json"))
        if isinstance(metrics_obj, dict):
            for key, value in metrics_obj.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    rec[f"metric.{key}"] = float(value)
        if isinstance(params_obj, dict):
            for key, value in params_obj.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    rec[f"param.{key}"] = float(value)
        analysis_rows.append(rec)
    return pd.DataFrame(analysis_rows)
