from __future__ import annotations

from datetime import date

import pandas as pd

from loto_forecast.api.streamlit import (
    dashboard_nf_resource_analytics_panel as panel,
)
from loto_forecast.api.streamlit import (
    dashboard_nf_resource_analytics_panel_aggregator as aggregator,
)
from loto_forecast.api.streamlit import (
    dashboard_nf_resource_analytics_panel_formatter as formatter,
)
from loto_forecast.api.streamlit import (
    dashboard_nf_resource_analytics_panel_helpers as helpers,
)
from loto_forecast.api.streamlit import (
    dashboard_nf_resource_analytics_panel_state as state,
)


def _parse_json_like(raw: object) -> dict[str, object]:
    return panel._parse_json_like(raw)


def _to_num(series: pd.Series) -> pd.Series:
    return panel._to_num(series)


def test_merge_run_model_metadata_and_filter_runs() -> None:
    run_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "status": ["success", "failed"],
            "app_name": ["train", "predict"],
            "command": ["python train", "python predict"],
            "duration_sec": [10.0, 20.0],
            "rows_written": [100, 50],
            "rows_failed": [0, 10],
        }
    )
    model_df = pd.DataFrame(
        {
            "run_id": ["r1"],
            "model_name": ["AutoNHITS"],
            "model_status": ["success"],
            "params_json": ['{"backend": "optuna", "search_alg": "TPESampler"}'],
        }
    )
    merged = helpers.merge_run_model_metadata(run_df, model_df, parse_json_like=_parse_json_like)
    assert merged.loc[0, "backend"] == "optuna"
    assert merged.loc[0, "search_alg"] == "TPESampler"

    filtered = helpers.filter_runs(
        merged,
        start_day=date(2026, 3, 1),
        end_day=date(2026, 3, 1),
        status_sel=["success"],
        app_sel=["train"],
        model_sel=["AutoNHITS"],
        command_kw="train",
    )
    assert filtered["run_id"].tolist() == ["r1"]


def test_normalize_run_metrics_baseline_and_anomaly_pipeline() -> None:
    run_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
            "status": ["success", "failed", "success"],
            "duration_sec": [10.0, 50.0, 12.0],
            "rows_written": [100.0, 10.0, 110.0],
            "rows_failed": [0.0, 40.0, 0.0],
            "group_key": ["A", "A", "A"],
            "db_time_ms": [1000.0, 40000.0, 1200.0],
            "gpu_util_avg": [70.0, 10.0, 60.0],
            "gpu_mem_used_mb_avg": [500.0, 600.0, 550.0],
        }
    )
    filtered = helpers.normalize_run_metrics(run_df, to_num=_to_num)
    baseline_pool = helpers.build_baseline_pool(
        run_df,
        filtered,
        baseline_mode="直近N件",
        baseline_n=3,
        baseline_days=None,
        end_day=date(2026, 3, 3),
        group_col="group_key",
        to_num=_to_num,
    )
    stats = helpers.build_baseline_stats(baseline_pool, group_col="group_key", to_num=_to_num)
    enriched = helpers.apply_expected_baselines(
        filtered,
        baseline_pool=baseline_pool,
        baseline_stats=stats,
        group_col="group_key",
        to_num=_to_num,
    )
    anomalous = helpers.apply_anomaly_detection(
        enriched,
        group_col="group_key",
        group_robust_zscore=panel._group_robust_zscore,
        group_iqr_high_flag=panel._group_iqr_high_flag,
    )
    assert "expected_duration_p50" in anomalous.columns
    assert "anomaly_flag" in anomalous.columns
    assert bool(anomalous.sort_values("duration_sec", ascending=False).iloc[0]["anomaly_flag"]) is True


def test_stage_and_event_aggregates_and_context() -> None:
    stage_df = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "stage_name": ["load", "train", "load"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-01", "2026-03-02"]),
            "duration_ms": [100.0, 300.0, 50.0],
            "db_time_ms": [10.0, 100.0, 5.0],
            "gpu_util_avg": [20.0, 80.0, 10.0],
            "gpu_mem_used_mb_avg": [100.0, 400.0, 50.0],
            "exception_type": [None, "ValueError", None],
        }
    )
    normalized_stage = helpers.normalize_stage_frame(stage_df, run_ids={"r1"}, to_num=_to_num)
    assert normalized_stage["run_id"].unique().tolist() == ["r1"]

    run_stage_agg = helpers.build_run_stage_agg(normalized_stage)
    assert run_stage_agg.iloc[0]["dominant_stage"] == "train"

    stage_agg = helpers.build_stage_aggregate_summary(normalized_stage, to_num=_to_num)
    assert "stage_share" in stage_agg.columns
    assert stage_agg.iloc[0]["stage_name"] == "train"

    hist_df = pd.DataFrame(
        {
            "run_id": ["r1", "r1"],
            "event_ts": pd.to_datetime(["2026-03-01T10:00:00", "2026-03-01T10:05:00"]),
            "event_type": ["start", "train"],
            "status": ["running", "running"],
        }
    )
    error_df = pd.DataFrame(
        {
            "run_id": ["r1"],
            "event_ts": pd.to_datetime(["2026-03-01T10:06:00"]),
            "error_type": ["ValueError"],
            "error_message": ["boom"],
        }
    )
    hist_agg = helpers.build_history_agg(hist_df, safe_mode=panel._safe_mode)
    err_agg = helpers.build_error_agg(error_df, safe_mode=panel._safe_mode)
    ctx_df = helpers.build_error_context(error_df, hist_df)
    assert hist_agg.iloc[0]["history_events"] == 2
    assert err_agg.iloc[0]["error_events"] == 1
    assert ctx_df.iloc[0]["prev_event_type"] == "train"


def test_panel_uses_resource_analytics_helpers() -> None:
    source = open(panel.__file__, encoding="utf-8").read()
    assert "helpers.merge_run_model_metadata" in source
    assert "panel_state.build_fetch_limits" in source
    assert "panel_formatter.build_summary_metrics" in source
    assert "panel_aggregator.merge_run_aggregates" in source


def test_resource_analytics_submodules_expose_expected_paths() -> None:
    assert state.build_fetch_limits(10)["run_fetch_limit"] == 500
    assert formatter.build_stage_db_rank(pd.DataFrame()).empty
    assert aggregator.build_selected_metric_frame(pd.DataFrame(), sel_run="r1", sel_key="gpu").empty
