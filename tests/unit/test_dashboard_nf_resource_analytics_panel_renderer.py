from __future__ import annotations

import pandas as pd
from _streamlit_test_double import FakeStreamlit

from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel as panel


def test_resource_analytics_renderer_delegates_through_helpers(monkeypatch) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(panel, "st", fake_st)

    run_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "ended_at": pd.to_datetime(["2026-03-01T00:01:00", "2026-03-02T00:02:00"]),
            "status": ["success", "failed"],
            "app_name": ["train", "train"],
            "command": ["python train", "python train"],
            "rows_target": [100, 100],
            "rows_written": [100, 50],
            "rows_failed": [0, 20],
            "error_summary": [None, "boom"],
            "tags": [{}, {}],
            "execution_os": ["linux", "linux"],
            "duration_sec": [60.0, 120.0],
        }
    )
    model_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "model_name": ["AutoNHITS", "AutoNHITS"],
            "model_status": ["success", "failed"],
            "params_json": ['{"backend":"optuna","search_alg":"TPESampler"}'] * 2,
            "created_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
        }
    )
    stage_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "stage_name": ["fit", "fit"],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "ended_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "duration_ms": [1000.0, 9000.0],
            "rows_in": [100, 100],
            "rows_out": [100, 50],
            "db_time_ms": [100.0, 5000.0],
            "db_rows": [10, 10],
            "gpu_util_avg": [70.0, 10.0],
            "gpu_mem_used_mb_avg": [500.0, 400.0],
            "exception_type": [None, "ValueError"],
            "exception_msg": [None, "boom"],
        }
    )
    metric_df = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "sampled_at": pd.to_datetime(["2026-03-01", "2026-03-01T00:00:30", "2026-03-02"], format="mixed"),
            "metric_key": ["gpu", "gpu", "gpu"],
            "metric_value": [70.0, 75.0, 20.0],
            "unit": ["pct", "pct", "pct"],
        }
    )
    hist_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "event_ts": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "event_type": ["start", "fit"],
            "status": ["running", "running"],
            "model_name": ["AutoNHITS", "AutoNHITS"],
            "dataset_name": ["d", "d"],
            "message": ["m1", "m2"],
        }
    )
    error_df = pd.DataFrame(
        {
            "run_id": ["r2"],
            "event_ts": pd.to_datetime(["2026-03-02"]),
            "model_name": ["AutoNHITS"],
            "stage": ["fit"],
            "error_type": ["ValueError"],
            "error_message": ["boom"],
        }
    )
    metric_def_df = pd.DataFrame({"metric_key": ["gpu"], "scope": ["run"], "unit": ["pct"]})

    def query_df(_engine, sql: str, params=None):
        if "FROM resources.run" in sql:
            return run_df.copy()
        if "FROM model.nf_automodel" in sql:
            return model_df.copy()
        if "FROM resources.stage_span" in sql:
            return stage_df.copy()
        if "FROM resources.resource_metric" in sql:
            return metric_df.copy()
        if "FROM log.run_history" in sql:
            return hist_df.copy()
        if "FROM log.error_event" in sql:
            return error_df.copy()
        if "FROM resources.metric_def" in sql:
            return metric_def_df.copy()
        raise AssertionError(sql)

    shown: list[pd.DataFrame] = []
    panel.render_nf_resource_analytics_panel(
        engine=object(),
        tables={
            ("resources", "run"),
            ("resources", "stage_span"),
            ("resources", "resource_metric"),
            ("resources", "metric_def"),
            ("log", "run_history"),
            ("log", "error_event"),
            ("model", "nf_automodel"),
        },
        row_limit=20,
        query_df=query_df,
        show_df=lambda df, **_: shown.append(df.copy()),
        plotly_available=False,
        px=None,
    )

    assert shown
    assert any(kind == "download_button" for kind, _ in fake_st.captured) is False
    assert any(kind == "tabs" for kind, _ in fake_st.captured)
