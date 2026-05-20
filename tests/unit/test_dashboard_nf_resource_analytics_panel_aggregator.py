from __future__ import annotations

from datetime import date

import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel as panel
from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_aggregator as aggregator


def test_merge_run_aggregates_and_baseline_pipeline() -> None:
    filtered = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "status": ["success", "failed"],
            "duration_sec": [10.0, 20.0],
            "rows_written": [100.0, 50.0],
            "rows_failed": [0.0, 5.0],
            "group_key": ["A", "A"],
        }
    )
    stage_df = pd.DataFrame(
        {
            "run_id": ["r1"],
            "stage_name": ["fit"],
            "duration_ms": [100.0],
            "db_time_ms": [10.0],
            "gpu_util_avg": [50.0],
            "gpu_mem_used_mb_avg": [500.0],
            "exception_type": [None],
        }
    )
    hist_df = pd.DataFrame({"run_id": ["r1"], "event_ts": pd.to_datetime(["2026-03-01"]), "event_type": ["start"]})
    error_df = pd.DataFrame({"run_id": ["r2"], "event_ts": pd.to_datetime(["2026-03-02"]), "error_type": ["boom"]})
    merged = aggregator.merge_run_aggregates(
        filtered,
        stage_df=stage_df,
        hist_df=hist_df,
        error_df=error_df,
        safe_mode=panel._safe_mode,
        to_num=panel._to_num,
    )
    assert "dominant_stage" in merged.columns
    assert "error_events" in merged.columns

    pool, stats, enriched = aggregator.apply_baseline_pipeline(
        filtered,
        merged,
        baseline_mode="直近N件",
        baseline_n=2,
        baseline_days=None,
        end_day=date(2026, 3, 2),
        group_col="group_key",
        to_num=panel._to_num,
    )
    assert not pool.empty
    assert not stats.empty
    assert "expected_duration_p50" in enriched.columns


def test_anomaly_gpu_metric_and_comparison_helpers() -> None:
    filtered = pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "duration_sec": [10.0, 50.0, 11.0],
            "throughput": [10.0, 1.0, 9.5],
            "duration_ratio_vs_expected": [1.0, 2.0, 1.0],
            "group_key": ["A", "A", "A"],
            "gpu_util_avg": [60.0, 10.0, 55.0],
            "sampled_at": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
        }
    )
    anomalous = aggregator.build_anomaly_enriched(
        filtered,
        group_col="group_key",
        group_robust_zscore=panel._group_robust_zscore,
        group_iqr_high_flag=panel._group_iqr_high_flag,
    )
    assert "anomaly_flag" in anomalous.columns

    gpu_idle = aggregator.build_gpu_idle_candidates(filtered, to_num=panel._to_num)
    assert gpu_idle.iloc[0]["run_id"] == "r2"

    metric_df = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "metric_key": ["gpu", "cpu", "gpu"],
            "sampled_at": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
            "metric_value": [1.0, 2.0, 3.0],
        }
    )
    selected = aggregator.build_selected_metric_frame(metric_df, sel_run="r1", sel_key="cpu")
    assert selected["metric_key"].tolist() == ["cpu"]

    sub = pd.DataFrame({"group_key": ["A", "A", "B"], "duration_sec": [1.0, 2.0, 3.0]})
    a, b = aggregator.build_comparison_arrays(
        sub,
        group_col="group_key",
        comp_metric="duration_sec",
        sel_groups=["A", "B"],
        to_num=panel._to_num,
    )
    assert a.tolist() == [1.0, 2.0]
    assert b.tolist() == [3.0]
