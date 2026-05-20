from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def build_fetch_limits(row_limit: int) -> dict[str, int]:
    return {
        "run_fetch_limit": int(max(500, min(30000, int(row_limit) * 20))),
        "stage_fetch_limit": int(max(3000, min(200000, int(row_limit) * 220))),
        "metric_fetch_limit": int(max(5000, min(250000, int(row_limit) * 450))),
        "log_fetch_limit": int(max(3000, min(150000, int(row_limit) * 300))),
    }


def resolve_date_bounds(run_df: pd.DataFrame) -> tuple[date, date, date]:
    valid_start = run_df["started_at"].dropna()
    if valid_start.empty:
        max_day = pd.Timestamp.utcnow().date()
        min_day = max_day
    else:
        max_day = valid_start.max().date()
        min_day = valid_start.min().date()
    default_start = max(min_day, (pd.Timestamp(max_day) - pd.Timedelta(days=30)).date())
    return min_day, max_day, default_start


def normalize_date_range(start_day: date, end_day: date) -> tuple[date, date]:
    if start_day > end_day:
        return end_day, start_day
    return start_day, end_day


def build_filter_options(run_df: pd.DataFrame) -> dict[str, list[str]]:
    status_opts = sorted(run_df["status"].dropna().astype(str).unique().tolist()) if "status" in run_df.columns else []
    app_opts = sorted(run_df["app_name"].dropna().astype(str).unique().tolist()) if "app_name" in run_df.columns else []
    model_opts = (
        sorted(run_df["model_name"].dropna().astype(str).unique().tolist()) if "model_name" in run_df.columns else []
    )
    group_candidates = [
        col
        for col in ["model_name", "backend", "search_alg", "app_name", "execution_os", "status"]
        if col in run_df.columns and run_df[col].notna().any()
    ]
    return {
        "status_opts": status_opts,
        "app_opts": app_opts,
        "model_opts": model_opts,
        "group_candidates": group_candidates or ["status"],
    }


def baseline_slider_config(run_count: int) -> dict[str, int]:
    max_n = max(50, min(10000, int(run_count)))
    return {"min_value": 20, "max_value": max_n, "value": min(max_n, 400), "step": 10}


def ensure_group_column(
    run_df: pd.DataFrame,
    filtered: pd.DataFrame,
    baseline_pool: pd.DataFrame,
    *,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_run = run_df.copy()
    out_filtered = filtered.copy()
    out_pool = baseline_pool.copy()
    if group_col not in out_run.columns:
        out_run[group_col] = "unknown"
    if group_col not in out_filtered.columns:
        out_filtered[group_col] = "unknown"
    else:
        out_filtered[group_col] = out_filtered[group_col].fillna("unknown")
    if group_col not in out_pool.columns:
        out_pool[group_col] = "unknown"
    else:
        out_pool[group_col] = out_pool[group_col].fillna("unknown")
    return out_run, out_filtered, out_pool


def build_slow_run_options(filtered: pd.DataFrame, *, limit: int = 100) -> list[str]:
    if filtered.empty:
        return []
    return filtered.sort_values("duration_sec", ascending=False)["run_id"].astype(str).head(limit).tolist()


def build_timeline_run_options(filtered: pd.DataFrame, *, limit: int = 200) -> list[str]:
    if filtered.empty:
        return []
    sort_cols = ["started_at"]
    ascending = [False]
    if "error_events" in filtered.columns:
        sort_cols.insert(0, "error_events")
        ascending.insert(0, False)
    return filtered.sort_values(sort_cols, ascending=ascending)["run_id"].astype(str).head(limit).tolist()


def build_metric_key_options(metric_df: pd.DataFrame) -> list[str]:
    if metric_df.empty:
        return []
    return (
        metric_df.groupby("metric_key", as_index=False)
        .size()
        .sort_values("size", ascending=False)["metric_key"]
        .astype(str)
        .tolist()
    )


def eligible_comparison_groups(comp_df: pd.DataFrame, *, group_col: str, min_size: int = 5) -> list[str]:
    if comp_df.empty:
        return []
    group_n = comp_df.groupby(group_col, as_index=False).size().sort_values("size", ascending=False)
    return group_n[group_n["size"] >= int(min_size)][group_col].astype(str).tolist()


def default_comparison_groups(groups: list[str]) -> list[str]:
    return groups[:2]


def panel_unavailable_message(*, engine: Any, tables: set[tuple[str, str]]) -> str | None:
    if engine is None:
        return "DB未接続のため利用できません。"
    if ("resources", "run") not in tables:
        return "resources.run が存在しないため利用できません。"
    return None
