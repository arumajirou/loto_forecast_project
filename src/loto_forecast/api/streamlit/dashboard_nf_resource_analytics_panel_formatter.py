from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_summary_metrics(filtered: pd.DataFrame, *, to_num: Any) -> list[dict[str, str]]:
    success_rate = (1.0 - float(filtered["is_failed"].mean())) * 100.0 if "is_failed" in filtered.columns else 0.0
    duration = filtered["duration_sec"] if "duration_sec" in filtered.columns else pd.Series(dtype=float)
    throughput = filtered["throughput"] if "throughput" in filtered.columns else pd.Series(dtype=float)
    fail_rate = filtered["fail_rate"] if "fail_rate" in filtered.columns else pd.Series(dtype=float)
    gpu = filtered.get("gpu_util_avg", pd.Series(dtype=float))
    db_share = filtered.get("db_share", pd.Series(dtype=float))
    return [
        {"label": "runs", "value": str(int(len(filtered)))},
        {"label": "success率", "value": f"{success_rate:.1f}%"},
        {
            "label": "duration p50(s)",
            "value": f"{float(duration.median()):.2f}" if duration.notna().any() else "n/a",
        },
        {
            "label": "duration p90(s)",
            "value": f"{float(duration.quantile(0.9)):.2f}" if duration.notna().any() else "n/a",
        },
        {
            "label": "throughput p50",
            "value": f"{float(to_num(throughput).median()):.2f}" if throughput.notna().any() else "n/a",
        },
        {
            "label": "fail_rate p90",
            "value": f"{float(to_num(fail_rate).quantile(0.9)):.1%}" if fail_rate.notna().any() else "n/a",
        },
        {
            "label": "GPU util p50",
            "value": f"{float(to_num(gpu).median()):.1f}%" if to_num(gpu).notna().any() else "n/a",
        },
        {
            "label": "DB share p50",
            "value": f"{float(to_num(db_share).median()):.1%}" if to_num(db_share).notna().any() else "n/a",
        },
    ]


def build_summary_columns(filtered: pd.DataFrame, *, group_col: str) -> list[str]:
    cols = [
        "run_id",
        "started_at",
        "status",
        group_col,
        "duration_sec",
        "expected_duration_p50",
        "duration_ratio_vs_expected",
        "rows_written",
        "rows_failed",
        "fail_rate",
        "throughput",
        "error_events",
        "anomaly_flag",
    ]
    return [col for col in cols if col in filtered.columns]


def build_rank_columns(filtered: pd.DataFrame, *, group_col: str) -> list[str]:
    cols = [
        "run_id",
        "started_at",
        "status",
        group_col,
        "duration_sec",
        "expected_duration_p50",
        "duration_ratio_vs_expected",
        "throughput",
        "expected_throughput_p50",
        "throughput_ratio_vs_expected",
        "fail_rate",
        "error_events",
        "dominant_stage",
        "db_share",
        "anomaly_score",
        "anomaly_flag",
    ]
    return [col for col in cols if col in filtered.columns]


def build_selected_stage_summary(sel_stage: pd.DataFrame, *, to_num: Any) -> pd.DataFrame:
    if sel_stage.empty:
        return pd.DataFrame()
    out = (
        sel_stage.groupby("stage_name", as_index=False)["duration_ms"]
        .sum()
        .sort_values("duration_ms", ascending=False)
    )
    total_sel_ms = float(to_num(out["duration_ms"]).sum())
    out["stage_share"] = np.where(total_sel_ms > 0, to_num(out["duration_ms"]) / total_sel_ms, np.nan)
    return out


def build_stage_db_rank(stage_agg: pd.DataFrame) -> pd.DataFrame:
    if stage_agg.empty or "db_share" not in stage_agg.columns:
        return pd.DataFrame()
    return stage_agg.sort_values("db_share", ascending=False)


def build_error_frequency_tables(error_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if error_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    err_type = error_df.groupby("error_type", as_index=False).size().sort_values("size", ascending=False)
    err_stage = error_df.groupby("stage", as_index=False).size().sort_values("size", ascending=False)
    return err_type, err_stage


def build_comparison_aggregate(sub: pd.DataFrame, *, group_col: str, comp_metric: str) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(group_col, as_index=False)[comp_metric]
        .agg(
            n="count",
            mean="mean",
            std="std",
            p50="median",
            p90=lambda x: float(pd.Series(x).quantile(0.9)),
        )
        .sort_values("n", ascending=False)
    )


def build_stat_payload(
    *,
    group_a: str,
    group_b: str,
    metric: str,
    a: np.ndarray,
    b: np.ndarray,
    cohen_d: Any,
    scipy_available: bool,
    spstats: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "group_a": group_a,
        "group_b": group_b,
        "metric": metric,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(np.mean(a)) if a.size > 0 else None,
        "mean_b": float(np.mean(b)) if b.size > 0 else None,
        "cohens_d": cohen_d(a, b),
    }
    if scipy_available and spstats is not None and a.size >= 2 and b.size >= 2:
        try:
            welch = spstats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            payload["welch_t_stat"] = float(welch.statistic)
            payload["welch_p_value"] = float(welch.pvalue)
        except Exception:
            pass
        try:
            mwu = spstats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
            payload["mannwhitney_u_stat"] = float(mwu.statistic)
            payload["mannwhitney_p_value"] = float(mwu.pvalue)
        except Exception:
            pass
    return payload
