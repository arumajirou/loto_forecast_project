from __future__ import annotations

from datetime import date

import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_resource_analytics_panel_state as state


def test_build_fetch_limits_and_date_bounds() -> None:
    limits = state.build_fetch_limits(25)
    assert limits["run_fetch_limit"] == 500
    assert limits["stage_fetch_limit"] == 5500

    run_df = pd.DataFrame({"started_at": pd.to_datetime(["2026-03-01", "2026-03-15", None])})
    min_day, max_day, default_start = state.resolve_date_bounds(run_df)
    assert min_day == date(2026, 3, 1)
    assert max_day == date(2026, 3, 15)
    assert default_start == date(2026, 3, 1)


def test_build_filter_options_group_columns_and_defaults() -> None:
    run_df = pd.DataFrame(
        {
            "status": ["success", "failed"],
            "app_name": ["train", "predict"],
            "model_name": ["A", "B"],
            "backend": ["optuna", None],
        }
    )
    options = state.build_filter_options(run_df)
    assert options["status_opts"] == ["failed", "success"]
    assert options["app_opts"] == ["predict", "train"]
    assert options["model_opts"] == ["A", "B"]
    assert "backend" in options["group_candidates"]

    fallback = state.build_filter_options(pd.DataFrame({"status": [None]}))
    assert fallback["group_candidates"] == ["status"]


def test_normalization_and_selection_helpers() -> None:
    assert state.normalize_date_range(date(2026, 3, 5), date(2026, 3, 1)) == (
        date(2026, 3, 1),
        date(2026, 3, 5),
    )
    assert state.baseline_slider_config(15)["max_value"] == 50

    filtered = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "duration_sec": [5.0, 10.0],
            "started_at": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "error_events": [0, 2],
        }
    )
    assert state.build_slow_run_options(filtered) == ["r2", "r1"]
    assert state.build_timeline_run_options(filtered) == ["r2", "r1"]

    metric_df = pd.DataFrame({"metric_key": ["gpu", "gpu", "cpu"]})
    assert state.build_metric_key_options(metric_df) == ["gpu", "cpu"]

    comp_df = pd.DataFrame({"group_key": ["A"] * 5 + ["B"] * 4 + ["C"] * 6, "value": list(range(15))})
    eligible = state.eligible_comparison_groups(comp_df, group_col="group_key")
    assert eligible == ["C", "A"]
    assert state.default_comparison_groups(eligible) == ["C", "A"]


def test_ensure_group_column_and_unavailable_message() -> None:
    run_df = pd.DataFrame({"run_id": ["r1"]})
    filtered = pd.DataFrame({"run_id": ["r1"], "status": [None]})
    baseline_pool = pd.DataFrame({"run_id": ["r1"]})
    out_run, out_filtered, out_pool = state.ensure_group_column(
        run_df,
        filtered,
        baseline_pool,
        group_col="status",
    )
    assert out_run["status"].tolist() == ["unknown"]
    assert out_filtered["status"].tolist() == ["unknown"]
    assert out_pool["status"].tolist() == ["unknown"]

    assert state.panel_unavailable_message(engine=None, tables=set()) == "DB未接続のため利用できません。"
    assert state.panel_unavailable_message(engine=object(), tables=set()) == "resources.run が存在しないため利用できません。"
    assert state.panel_unavailable_message(engine=object(), tables={("resources", "run")}) is None
