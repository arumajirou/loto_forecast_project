from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def build_overview_metrics(
    *,
    sel_run: str,
    sel_meta: dict[str, Any],
    sel_model_row: dict[str, Any],
    sel_dir: Path,
    sel_eval: dict[str, Any],
    settings: Any,
    has_model_artifacts: Any,
) -> list[dict[str, str]]:
    return [
        {"label": "run_id", "value": str(sel_run)},
        {"label": "model", "value": str(sel_model_row.get("model_name") or sel_meta.get("model_name") or "-")},
        {"label": "h", "value": str(int(sel_meta.get("h") or sel_model_row.get("horizon") or settings.default_horizon))},
        {"label": "status", "value": str(sel_model_row.get("status") or "-")},
        {"label": "artifact dir", "value": "yes" if sel_dir.exists() else "no"},
        {"label": "model files", "value": "yes" if has_model_artifacts(sel_dir) else "no"},
        {"label": "forecast.parquet", "value": "yes" if (sel_dir / "forecast.parquet").exists() else "no"},
        {"label": "evaluation.json", "value": "yes" if bool(sel_eval) else "no"},
    ]


def build_resource_metrics(rr: pd.Series) -> list[dict[str, str]]:
    return [
        {"label": "duration_sec", "value": f"{float(rr.get('duration_sec', 0.0) or 0.0):.2f}"},
        {"label": "rows_written", "value": str(int(rr.get("rows_written", 0) or 0))},
        {"label": "rows_failed", "value": str(int(rr.get("rows_failed", 0) or 0))},
        {"label": "run_status", "value": str(rr.get("status", "-"))},
    ]


def build_accuracy_aggregate(use_df: pd.DataFrame) -> pd.DataFrame:
    if use_df.empty:
        return pd.DataFrame()
    return (
        use_df.groupby("model_name", as_index=False)
        .agg(
            samples=("value", "count"),
            mean_value=("value", "mean"),
            p50=("value", "median"),
            p90=("value", lambda x: float(pd.Series(x).quantile(0.9))),
        )
        .sort_values("mean_value")
    )


def build_export_payload(
    *,
    sel_run: str,
    sel_meta: dict[str, Any],
    sel_eval: dict[str, Any],
    sel_model_row: dict[str, Any],
    sel_params: dict[str, Any],
    sel_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": str(sel_run),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": sel_meta,
        "evaluation": sel_eval,
        "model_row": sel_model_row,
        "params_json": sel_params,
        "metrics_json": sel_metrics,
    }


def build_export_preview(payload: dict[str, Any], *, stable_json_dumps: Any) -> str:
    return stable_json_dumps(payload)[:4000]
