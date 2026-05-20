from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel as panel
from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_formatter as formatter


def test_summary_and_rank_columns_and_metrics() -> None:
    filtered = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "status": ["success", "failed"],
            "group_key": ["A", "A"],
            "duration_sec": [10.0, 20.0],
            "rows_written": [100.0, 50.0],
            "rows_failed": [0.0, 10.0],
            "fail_rate": [0.0, 0.2],
            "throughput": [10.0, 2.5],
            "is_failed": [False, True],
            "gpu_util_avg": [40.0, 20.0],
            "db_share": [0.1, 0.5],
        }
    )
    metrics = formatter.build_summary_metrics(filtered, to_num=panel._to_num)
    assert metrics[0] == {"label": "runs", "value": "2"}
    assert metrics[1]["value"] == "50.0%"

    summary_cols = formatter.build_summary_columns(filtered, group_col="group_key")
    rank_cols = formatter.build_rank_columns(filtered, group_col="group_key")
    assert "group_key" in summary_cols
    assert "group_key" in rank_cols


def test_stage_error_and_comparison_formatters() -> None:
    sel_stage = pd.DataFrame({"stage_name": ["load", "train"], "duration_ms": [100.0, 300.0]})
    ssum = formatter.build_selected_stage_summary(sel_stage, to_num=panel._to_num)
    assert ssum.iloc[0]["stage_name"] == "train"
    assert float(ssum["stage_share"].sum()) == 1.0

    stage_agg = pd.DataFrame({"stage_name": ["a", "b"], "db_share": [0.1, 0.9]})
    assert formatter.build_stage_db_rank(stage_agg)["stage_name"].tolist() == ["b", "a"]

    error_df = pd.DataFrame({"error_type": ["ValueError", "ValueError", "KeyError"], "stage": ["fit", "fit", "load"]})
    err_type, err_stage = formatter.build_error_frequency_tables(error_df)
    assert err_type.iloc[0]["error_type"] == "ValueError"
    assert set(err_stage["stage"]) == {"fit", "load"}

    sub = pd.DataFrame({"group_key": ["A", "A", "B", "B"], "duration_sec": [1.0, 2.0, 3.0, 4.0]})
    agg = formatter.build_comparison_aggregate(sub, group_col="group_key", comp_metric="duration_sec")
    assert agg.iloc[0]["n"] == 2

    fake_stats = SimpleNamespace(
        ttest_ind=lambda *args, **kwargs: SimpleNamespace(statistic=1.2, pvalue=0.3),
        mannwhitneyu=lambda *args, **kwargs: SimpleNamespace(statistic=2.0, pvalue=0.4),
    )
    payload = formatter.build_stat_payload(
        group_a="A",
        group_b="B",
        metric="duration_sec",
        a=np.array([1.0, 2.0]),
        b=np.array([3.0, 4.0]),
        cohen_d=panel._cohen_d,
        scipy_available=True,
        spstats=fake_stats,
    )
    assert payload["group_a"] == "A"
    assert payload["welch_p_value"] == 0.3
