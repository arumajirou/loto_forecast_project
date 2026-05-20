from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_runid_panel as panel
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_analysis as analysis
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_formatter as formatter
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_helpers as helpers
from loto_forecast.api.streamlit import dashboard_nf_runid_panel_state as state
from loto_forecast.api.streamlit import operations_dashboard_helpers as operations_helpers


def test_build_run_snapshot_and_config_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "configuration.pkl").write_text("x", encoding="utf-8")
    (run_dir / "alias_to_model.pkl").write_text("x", encoding="utf-8")
    sel_model = pd.DataFrame([{"run_id": "r1", "model_name": "AutoNHITS", "horizon": 7}])
    snapshot = helpers.build_run_snapshot(
        sel_run="r1",
        sel_dir=run_dir,
        sel_meta={"model_name": "AutoNHITS", "h": 7, "nf_runtime_kwargs": {}},
        sel_model=sel_model,
        sel_model_row={"model_name": "AutoNHITS", "horizon": 7},
        settings=SimpleNamespace(default_horizon=14),
        has_model_artifacts=lambda path: True,
    )
    assert snapshot["artifact_exists"] is True
    assert snapshot["model_files"] is True

    checks = helpers.build_config_check_rows(
        meta_model_name="AutoNHITS",
        db_model_name="AutoNHITS",
        meta_h=7,
        db_h=7,
        expected_backend="optuna",
        actual_backend="optuna",
        expected_num_samples=10,
        actual_num_samples=10,
        meta_pred_h=7,
        meta_cv_h=7,
        safe_int_eq=panel._safe_int_eq,
    )
    assert len(checks) == 6
    assert all(row["ok"] is True for row in checks if row["ok"] is not None)


def test_build_model_resource_metric_and_analysis_frames() -> None:
    model_resource_df = pd.DataFrame(
        [{"model_name": "A", "avg_rows_written": 100.0, "avg_duration_sec": 10.0, "run_count": 3}]
    )
    shaped = helpers.build_model_resource_df(model_resource_df)
    assert shaped.iloc[0]["efficiency_score"] == 10.0

    model_df = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "model_name": "A",
                "status": "success",
                "horizon": 7,
                "metrics_json": '{"mae": 1.2, "rmse": 2.4}',
                "params_json": '{"num_samples": 10, "backend": "optuna"}',
            }
        ]
    )
    metric_df = helpers.build_metric_rows(model_df, parse_json_like=operations_helpers.parse_json_like, row_limit=10)
    analysis_df = helpers.build_analysis_df(model_df, parse_json_like=operations_helpers.parse_json_like, row_limit=10)
    assert set(metric_df["metric"]) == {"mae", "rmse"}
    assert analysis_df.iloc[0]["metric.mae"] == 1.2
    assert analysis_df.iloc[0]["param.num_samples"] == 10.0


def test_runid_panel_uses_helper_calls() -> None:
    source = open(panel.__file__, encoding="utf-8").read()
    assert "helpers.build_run_snapshot" in source
    assert "helpers.build_config_check_rows" in source
    assert "helpers.build_metric_rows" in source
    assert "helpers.build_analysis_df" in source
    assert "panel_state.resolve_selected_model" in source
    assert "panel_formatter.build_accuracy_aggregate" in source
    assert "panel_analysis.build_correlation_rows" in source


def test_runid_panel_submodules_basic_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "artifact.bin").write_text("x", encoding="utf-8")
    files = state.build_file_rows(run_dir)
    assert files.iloc[0]["file"] == "artifact.bin"
    assert state.default_target_metric(["metric.rmse"]) == "metric.rmse"
    assert analysis.build_correlation_rows(pd.DataFrame(), target_col="metric.rmse").empty
    assert formatter.build_accuracy_aggregate(pd.DataFrame()).empty
