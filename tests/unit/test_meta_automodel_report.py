from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from loto_forecast.analysis.meta_automodel_report import build_meta_automodel_report


def _sample_raw_df(n: int = 12) -> pd.DataFrame:
    base = datetime(2026, 2, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        created = base + timedelta(hours=i)
        started = created + timedelta(minutes=1)
        ended = started + timedelta(minutes=2 + (i % 3))
        seed = 1 if i % 2 == 0 else 2
        num_samples = 10 if i % 3 == 0 else 20
        lr = 0.01 + (0.002 * (i % 5))
        model_name = "AutoNHITS" if i % 2 == 0 else "AutoNBEATS"
        mae = 0.15 + (0.02 * (i % 4)) + (0.03 if seed == 2 else 0.0)
        mae += 0.06 if num_samples == 10 else 0.0
        mae += 0.4 * lr
        rmse = mae + 0.1
        status = "failed" if i in {3, 9} else "success"
        rows.append(
            {
                "config_id": 2,
                "run_id": f"run_{i:03d}",
                "status": status,
                "model_name": model_name,
                "horizon": 28,
                "cfg_horizon": 28,
                "cfg_model_name": model_name,
                "cfg_auto_num_samples": (99 if i == 5 else num_samples),
                "cfg_auto_backend": "optuna",
                "created_at": created,
                "started_at": started,
                "ended_at": ended,
                "params_json": {
                    "seed": seed,
                    "num_samples": num_samples,
                    "backend": "optuna",
                    "h": 28,
                    "learning_rate": lr,
                },
                "metrics_json": {
                    "mae": mae,
                    "rmse": rmse,
                },
                "diagnostics_json": {
                    "lb_pvalue": 0.01 if i % 5 == 0 else 0.3,
                    "adf_pvalue": 0.2 if i % 4 == 0 else 0.04,
                },
                "explain_json": {"feature_count": 12 + i},
            }
        )
    return pd.DataFrame(rows)


def test_build_meta_automodel_report_basic_sections():
    df = _sample_raw_df()
    out = build_meta_automodel_report(
        raw_df=df,
        target_metric="mae",
        recursive_depth=2,
        min_group_size=2,
        alpha=0.05,
        top_k=10,
    )

    assert out["target_metric"] == "metrics.mae"
    assert out["overview"]["rows"] == 12
    assert not out["metric_summary"].empty
    assert not out["parameter_reflection"].empty
    assert "normality_jarque_bera" in out["stat_tests"]
    assert "status_mean_diff_test" in out["stat_tests"]
    assert "model_name_kruskal" in out["stat_tests"]
    assert isinstance(out["recursive_tree"], dict)
    assert out["recursive_tree"]["depth"] == 0
    assert len(out["insights"]) >= 1


def test_build_meta_automodel_report_parameter_impact_present():
    df = _sample_raw_df()
    out = build_meta_automodel_report(
        raw_df=df,
        target_metric="metrics.mae",
        recursive_depth=3,
        min_group_size=2,
    )
    impact = out["parameter_impact"]
    assert not impact.empty
    assert "parameter" in impact.columns
    assert "effect_abs" in impact.columns
    assert "feature_contribution" in out
    assert not out["feature_contribution"].empty
    assert "causal_hints" in out
    assert isinstance(out["causal_hints"], pd.DataFrame)


def test_build_meta_automodel_report_no_metrics_graceful():
    df = _sample_raw_df()
    df["metrics_json"] = [{} for _ in range(len(df))]
    out = build_meta_automodel_report(raw_df=df, target_metric="metrics.mae")
    assert out["target_metric"] is None
    assert "error" in out["stat_tests"]
    assert "parameter_reflection" in out


def test_build_meta_automodel_report_no_params_does_not_crash():
    df = _sample_raw_df()
    df["params_json"] = [{} for _ in range(len(df))]
    out = build_meta_automodel_report(raw_df=df, target_metric="metrics.mae", recursive_depth=3)
    assert out["target_metric"] == "metrics.mae"
    assert out["parameter_impact"].empty
    assert isinstance(out["recursive_tree"], dict)


def test_build_meta_automodel_report_reflection_detects_mismatch():
    df = _sample_raw_df()
    out = build_meta_automodel_report(raw_df=df, target_metric="metrics.mae", recursive_depth=3)
    reflection = out["parameter_reflection"]
    assert not reflection.empty
    row = reflection.loc[reflection["parameter"] == "auto_num_samples"]
    assert not row.empty
    assert int(row.iloc[0]["mismatched_rows"]) >= 1
