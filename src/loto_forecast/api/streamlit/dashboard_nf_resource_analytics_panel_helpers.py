from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


def merge_run_model_metadata(
    run_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    parse_json_like: Any,
) -> pd.DataFrame:
    if run_df.empty or model_df.empty:
        return run_df.copy()
    out = run_df.copy()
    model = model_df.drop_duplicates(subset=["run_id"], keep="first").copy()
    model["params_obj"] = model["params_json"].apply(parse_json_like)
    model["backend"] = model["params_obj"].apply(
        lambda obj: str(obj.get("backend")) if isinstance(obj, dict) and obj.get("backend") is not None else None
    )
    model["search_alg"] = model["params_obj"].apply(
        lambda obj: (
            str(obj.get("search_alg") or obj.get("search_algorithm"))
            if isinstance(obj, dict) and (obj.get("search_alg") or obj.get("search_algorithm")) is not None
            else None
        )
    )
    model = model[["run_id", "model_name", "model_status", "backend", "search_alg"]]
    return out.merge(model, on="run_id", how="left")


def filter_runs(
    run_df: pd.DataFrame,
    *,
    start_day: date,
    end_day: date,
    status_sel: list[str],
    app_sel: list[str],
    model_sel: list[str],
    command_kw: str,
) -> pd.DataFrame:
    filtered = run_df.copy()
    filtered = filtered[(filtered["started_at"].dt.date >= start_day) & (filtered["started_at"].dt.date <= end_day)]
    if status_sel:
        filtered = filtered[filtered["status"].astype(str).isin([str(x) for x in status_sel])]
    if app_sel:
        filtered = filtered[filtered["app_name"].astype(str).isin([str(x) for x in app_sel])]
    if model_sel and "model_name" in filtered.columns:
        filtered = filtered[filtered["model_name"].astype(str).isin([str(x) for x in model_sel])]
    if command_kw.strip():
        kw = command_kw.strip().lower()
        filtered = filtered[filtered["command"].fillna("").astype(str).str.lower().str.contains(kw, na=False)]
    return filtered.sort_values("started_at", ascending=False).copy()


def normalize_run_metrics(run_df: pd.DataFrame, *, to_num: Any) -> pd.DataFrame:
    out = run_df.copy()
    out["duration_sec"] = to_num(out["duration_sec"])
    out["rows_written"] = to_num(out["rows_written"]).fillna(0.0)
    out["rows_failed"] = to_num(out["rows_failed"]).fillna(0.0)
    out["rows_total"] = out["rows_written"] + out["rows_failed"]
    out["fail_rate"] = np.where(out["rows_total"] > 0, out["rows_failed"] / out["rows_total"], 0.0)
    out["throughput"] = np.where(out["duration_sec"] > 0, out["rows_written"] / out["duration_sec"], np.nan)
    out["is_failed"] = out["status"].astype(str).str.lower().eq("failed")
    if "db_time_ms" in out.columns:
        out["db_share"] = np.where(
            out["duration_sec"] > 0,
            to_num(out["db_time_ms"]) / (out["duration_sec"] * 1000.0),
            np.nan,
        )
    return out


def normalize_stage_frame(stage_df: pd.DataFrame, *, run_ids: set[str], to_num: Any) -> pd.DataFrame:
    if stage_df.empty:
        return stage_df.copy()
    out = stage_df.copy()
    out["run_id"] = out["run_id"].astype(str)
    out = out[out["run_id"].isin(run_ids)].copy()
    out["started_at"] = pd.to_datetime(out["started_at"], errors="coerce")
    for col in ["duration_ms", "db_time_ms", "gpu_util_avg", "gpu_mem_used_mb_avg"]:
        out[col] = to_num(out[col])
    return out


def normalize_metric_frame(metric_df: pd.DataFrame, *, run_ids: set[str], to_num: Any) -> pd.DataFrame:
    if metric_df.empty:
        return metric_df.copy()
    out = metric_df.copy()
    out["run_id"] = out["run_id"].astype(str)
    out = out[out["run_id"].isin(run_ids)].copy()
    out["sampled_at"] = pd.to_datetime(out["sampled_at"], errors="coerce")
    out["metric_value"] = to_num(out["metric_value"])
    return out


def normalize_event_frame(event_df: pd.DataFrame, *, run_ids: set[str], time_col: str) -> pd.DataFrame:
    if event_df.empty:
        return event_df.copy()
    out = event_df.copy()
    out["run_id"] = out["run_id"].astype(str)
    out = out[out["run_id"].isin(run_ids)].copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    return out


def build_run_stage_agg(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame()
    run_stage_agg = stage_df.groupby("run_id", as_index=False).agg(
        stage_duration_ms=("duration_ms", "sum"),
        db_time_ms=("db_time_ms", "sum"),
        gpu_util_avg=("gpu_util_avg", "mean"),
        gpu_mem_used_mb_avg=("gpu_mem_used_mb_avg", "mean"),
        stage_exception_count=("exception_type", lambda s: int(pd.Series(s).notna().sum())),
    )
    dominant_stage = (
        stage_df.groupby(["run_id", "stage_name"], as_index=False)["duration_ms"]
        .sum()
        .sort_values(["run_id", "duration_ms"], ascending=[True, False])
        .drop_duplicates(subset=["run_id"], keep="first")
        .rename(columns={"stage_name": "dominant_stage", "duration_ms": "dominant_stage_duration_ms"})
    )
    return run_stage_agg.merge(dominant_stage, on="run_id", how="left")


def build_history_agg(hist_df: pd.DataFrame, *, safe_mode: Any) -> pd.DataFrame:
    if hist_df.empty:
        return pd.DataFrame()
    return hist_df.groupby("run_id", as_index=False).agg(
        history_events=("event_type", "count"),
        last_event_ts=("event_ts", "max"),
        last_event_type=("event_type", safe_mode),
    )


def build_error_agg(error_df: pd.DataFrame, *, safe_mode: Any) -> pd.DataFrame:
    if error_df.empty:
        return pd.DataFrame()
    return error_df.groupby("run_id", as_index=False).agg(
        error_events=("error_type", "count"),
        last_error_ts=("event_ts", "max"),
        top_error_type=("error_type", safe_mode),
    )


def build_baseline_pool(
    run_df: pd.DataFrame,
    filtered: pd.DataFrame,
    *,
    baseline_mode: str,
    baseline_n: int | None,
    baseline_days: int | None,
    end_day: date,
    group_col: str,
    to_num: Any,
) -> pd.DataFrame:
    if baseline_mode == "直近N件":
        baseline_count = baseline_n if baseline_n is not None else min(len(run_df), 400)
        pool = run_df.sort_values("started_at", ascending=False).head(baseline_count).copy()
    else:
        start_base = pd.Timestamp(end_day) - pd.Timedelta(days=int(baseline_days or 60))
        pool = run_df[run_df["started_at"] >= start_base].copy()
    if pool.empty:
        pool = filtered.copy()
    pool = normalize_run_metrics(pool, to_num=to_num)
    if group_col not in pool.columns:
        pool[group_col] = "unknown"
    if "db_share" in filtered.columns and "db_share" not in pool.columns:
        ref = filtered[["run_id", "db_share"]].drop_duplicates(subset=["run_id"])
        pool = pool.merge(ref, on="run_id", how="left")
    if "gpu_util_avg" in filtered.columns and "gpu_util_avg" not in pool.columns:
        ref = filtered[["run_id", "gpu_util_avg", "gpu_mem_used_mb_avg"]].drop_duplicates(subset=["run_id"])
        pool = pool.merge(ref, on="run_id", how="left")
    return pool


def build_baseline_stats(baseline_pool: pd.DataFrame, *, group_col: str, to_num: Any) -> pd.DataFrame:
    agg_spec: dict[str, tuple[str, Any]] = {
        "baseline_n": ("run_id", "count"),
        "expected_duration_p50": ("duration_sec", "median"),
        "expected_duration_p90": ("duration_sec", lambda x: float(pd.Series(x).quantile(0.9))),
        "expected_throughput_p50": ("throughput", "median"),
        "expected_fail_rate_p50": ("fail_rate", "median"),
    }
    if "gpu_util_avg" in baseline_pool.columns:
        agg_spec["expected_gpu_util_p50"] = ("gpu_util_avg", "median")
    if "db_share" in baseline_pool.columns:
        agg_spec["expected_db_share_p50"] = ("db_share", "median")
    stats = baseline_pool.groupby(group_col, as_index=False).agg(**agg_spec)
    if "expected_gpu_util_p50" not in stats.columns:
        stats["expected_gpu_util_p50"] = np.nan
    if "expected_db_share_p50" not in stats.columns:
        stats["expected_db_share_p50"] = np.nan
    stats["expected_duration_p50"] = to_num(stats["expected_duration_p50"])
    stats["expected_throughput_p50"] = to_num(stats["expected_throughput_p50"])
    return stats


def apply_expected_baselines(
    filtered: pd.DataFrame,
    *,
    baseline_pool: pd.DataFrame,
    baseline_stats: pd.DataFrame,
    group_col: str,
    to_num: Any,
) -> pd.DataFrame:
    out = filtered.merge(baseline_stats, on=group_col, how="left")
    global_expected_duration = (
        float(to_num(baseline_pool["duration_sec"]).median()) if baseline_pool["duration_sec"].notna().any() else np.nan
    )
    global_expected_throughput = (
        float(to_num(baseline_pool["throughput"]).median()) if baseline_pool["throughput"].notna().any() else np.nan
    )
    out["expected_duration_p50"] = to_num(out["expected_duration_p50"]).fillna(global_expected_duration)
    out["expected_throughput_p50"] = to_num(out["expected_throughput_p50"]).fillna(global_expected_throughput)
    out["duration_vs_expected"] = out["duration_sec"] - out["expected_duration_p50"]
    out["duration_ratio_vs_expected"] = np.where(
        out["expected_duration_p50"] > 0,
        out["duration_sec"] / out["expected_duration_p50"],
        np.nan,
    )
    out["throughput_ratio_vs_expected"] = np.where(
        out["expected_throughput_p50"] > 0,
        out["throughput"] / out["expected_throughput_p50"],
        np.nan,
    )
    return out


def apply_anomaly_detection(
    filtered: pd.DataFrame,
    *,
    group_col: str,
    group_robust_zscore: Any,
    group_iqr_high_flag: Any,
) -> pd.DataFrame:
    out = filtered.copy()
    out["duration_rz"] = group_robust_zscore(out, "duration_sec", group_col)
    out["throughput_rz"] = group_robust_zscore(out, "throughput", group_col)
    out["duration_iqr_high"] = group_iqr_high_flag(out, "duration_sec", group_col)
    out["anomaly_score"] = (
        out["duration_rz"].abs().fillna(0.0)
        + out["throughput_rz"].abs().fillna(0.0)
        + np.where(out["duration_iqr_high"], 1.0, 0.0)
    )
    out["anomaly_flag"] = (
        (out["duration_rz"].abs() >= 3.0)
        | (out["throughput_rz"] <= -3.0)
        | out["duration_iqr_high"].fillna(False)
        | (out["duration_ratio_vs_expected"] >= 1.8)
    )
    return out


def build_stage_aggregate_summary(stage_df: pd.DataFrame, *, to_num: Any) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame()
    stage_agg = (
        stage_df.groupby("stage_name", as_index=False)
        .agg(
            count=("stage_name", "count"),
            total_duration_ms=("duration_ms", "sum"),
            avg_duration_ms=("duration_ms", "mean"),
            total_db_time_ms=("db_time_ms", "sum"),
            avg_gpu_util=("gpu_util_avg", "mean"),
            avg_gpu_mem_mb=("gpu_mem_used_mb_avg", "mean"),
            exception_count=("exception_type", lambda s: int(pd.Series(s).notna().sum())),
        )
        .sort_values("total_duration_ms", ascending=False)
    )
    total_ms = float(to_num(stage_agg["total_duration_ms"]).sum())
    stage_agg["stage_share"] = to_num(stage_agg["total_duration_ms"]) / total_ms if total_ms > 0 else np.nan
    stage_agg["cum_share"] = to_num(stage_agg["stage_share"]).fillna(0.0).cumsum()
    stage_agg["db_share"] = np.where(
        to_num(stage_agg["total_duration_ms"]) > 0,
        to_num(stage_agg["total_db_time_ms"]) / to_num(stage_agg["total_duration_ms"]),
        np.nan,
    )
    return stage_agg


def build_error_context(error_df: pd.DataFrame, hist_df: pd.DataFrame) -> pd.DataFrame:
    if error_df.empty or hist_df.empty:
        return pd.DataFrame()
    ctx_rows: list[dict[str, Any]] = []
    hist_group = {rid: h.sort_values("event_ts") for rid, h in hist_df.groupby("run_id")}
    for row in error_df.sort_values("event_ts").head(300).to_dict(orient="records"):
        rid = str(row.get("run_id", ""))
        event_ts = row.get("event_ts")
        hist = hist_group.get(rid)
        if hist is None or pd.isna(event_ts):
            ctx_rows.append({**row, "prev_event_ts": None, "prev_event_type": None, "prev_status": None})
            continue
        prev = hist[hist["event_ts"] <= event_ts].tail(1)
        if prev.empty:
            ctx_rows.append({**row, "prev_event_ts": None, "prev_event_type": None, "prev_status": None})
            continue
        prev_row = prev.iloc[0]
        ctx_rows.append(
            {
                **row,
                "prev_event_ts": prev_row.get("event_ts"),
                "prev_event_type": prev_row.get("event_type"),
                "prev_status": prev_row.get("status"),
            }
        )
    return pd.DataFrame(ctx_rows)
