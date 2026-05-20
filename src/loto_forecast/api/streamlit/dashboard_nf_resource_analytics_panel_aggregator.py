from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_helpers as helpers


def merge_run_aggregates(
    filtered: pd.DataFrame,
    *,
    stage_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    error_df: pd.DataFrame,
    safe_mode: Any,
    to_num: Any,
) -> pd.DataFrame:
    out = filtered.copy()
    if not stage_df.empty:
        run_stage_agg = helpers.build_run_stage_agg(stage_df)
        out = out.merge(run_stage_agg, on="run_id", how="left")
    if not hist_df.empty:
        hist_agg = helpers.build_history_agg(hist_df, safe_mode=safe_mode)
        out = out.merge(hist_agg, on="run_id", how="left")
    if not error_df.empty:
        err_agg = helpers.build_error_agg(error_df, safe_mode=safe_mode)
        out = out.merge(err_agg, on="run_id", how="left")
    else:
        out["error_events"] = 0
    return helpers.normalize_run_metrics(out, to_num=to_num)


def apply_baseline_pipeline(
    run_df: pd.DataFrame,
    filtered: pd.DataFrame,
    *,
    baseline_mode: str,
    baseline_n: int | None,
    baseline_days: int | None,
    end_day: Any,
    group_col: str,
    to_num: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_pool = helpers.build_baseline_pool(
        run_df,
        filtered,
        baseline_mode=baseline_mode,
        baseline_n=baseline_n,
        baseline_days=baseline_days,
        end_day=end_day,
        group_col=group_col,
        to_num=to_num,
    )
    baseline_stats = helpers.build_baseline_stats(baseline_pool, group_col=group_col, to_num=to_num)
    enriched = helpers.apply_expected_baselines(
        filtered,
        baseline_pool=baseline_pool,
        baseline_stats=baseline_stats,
        group_col=group_col,
        to_num=to_num,
    )
    return baseline_pool, baseline_stats, enriched


def build_anomaly_enriched(
    filtered: pd.DataFrame,
    *,
    group_col: str,
    group_robust_zscore: Any,
    group_iqr_high_flag: Any,
) -> pd.DataFrame:
    return helpers.apply_anomaly_detection(
        filtered,
        group_col=group_col,
        group_robust_zscore=group_robust_zscore,
        group_iqr_high_flag=group_iqr_high_flag,
    )


def build_gpu_idle_candidates(filtered: pd.DataFrame, *, to_num: Any) -> pd.DataFrame:
    if "gpu_util_avg" not in filtered.columns or filtered.empty:
        return pd.DataFrame()
    dur_q75 = float(to_num(filtered["duration_sec"]).quantile(0.75))
    return filtered[
        (to_num(filtered["duration_sec"]) >= dur_q75) & (to_num(filtered["gpu_util_avg"]) < 25)
    ].sort_values("duration_sec", ascending=False)


def build_selected_metric_frame(metric_df: pd.DataFrame, *, sel_run: str, sel_key: str) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame()
    out = metric_df[
        (metric_df["run_id"].astype(str) == str(sel_run)) & (metric_df["metric_key"].astype(str) == str(sel_key))
    ].sort_values("sampled_at")
    return out


def build_comparison_arrays(
    sub: pd.DataFrame,
    *,
    group_col: str,
    comp_metric: str,
    sel_groups: list[str],
    to_num: Any,
) -> tuple[np.ndarray, np.ndarray]:
    group_a, group_b = str(sel_groups[0]), str(sel_groups[1])
    a = to_num(sub[sub[group_col].astype(str) == group_a][comp_metric]).dropna().to_numpy(dtype=float)
    b = to_num(sub[sub[group_col].astype(str) == group_b][comp_metric]).dropna().to_numpy(dtype=float)
    return a, b
