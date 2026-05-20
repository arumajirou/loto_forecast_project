from __future__ import annotations

from typing import Any

import pandas as pd


def merge_resource_features(
    analysis_df: pd.DataFrame,
    *,
    run_feat: pd.DataFrame,
    stage_feat: pd.DataFrame,
) -> pd.DataFrame:
    out = analysis_df.copy()
    if not run_feat.empty and "run_id" in run_feat.columns:
        run_norm = run_feat.copy()
        run_norm["run_id"] = run_norm["run_id"].astype(str)
        out = out.merge(run_norm, on="run_id", how="left")
    if not stage_feat.empty and "run_id" in stage_feat.columns:
        stage_norm = stage_feat.copy()
        stage_norm["run_id"] = stage_norm["run_id"].astype(str)
        out = out.merge(stage_norm, on="run_id", how="left")
    return out


def feature_columns(analysis_base: pd.DataFrame) -> list[str]:
    return [
        col
        for col in analysis_base.columns
        if col.startswith("param.")
        or col
        in {
            "horizon",
            "run_duration_sec",
            "rows_written",
            "rows_failed",
            "gpu_util_avg",
            "gpu_mem_avg_mb",
            "stage_total_ms",
        }
    ]


def build_correlation_rows(analysis_base: pd.DataFrame, *, target_col: str, min_samples: int = 12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in feature_columns(analysis_base):
        sample_df = analysis_base[[col, target_col]].dropna()
        if sample_df.shape[0] < int(min_samples):
            continue
        rows.append(
            {
                "feature": col,
                "pearson": float(sample_df[col].corr(sample_df[target_col])),
                "spearman": float(sample_df[col].corr(sample_df[target_col], method="spearman")),
                "n": int(sample_df.shape[0]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
