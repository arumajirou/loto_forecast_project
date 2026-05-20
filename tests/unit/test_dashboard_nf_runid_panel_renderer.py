from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from _streamlit_test_double import FakeStreamlit

from loto_forecast.api.streamlit import dashboard_nf_runid_panel as panel


def test_runid_renderer_delegates_through_submodules(monkeypatch, tmp_path: Path) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(panel, "st", fake_st)

    run_dir = tmp_path / "artifacts" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "forecast.parquet").write_text("x", encoding="utf-8")
    (run_dir / "configuration.pkl").write_text("x", encoding="utf-8")
    (run_dir / "alias_to_model.pkl").write_text("x", encoding="utf-8")

    model_df = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "model_name": "AutoNHITS",
                "status": "success",
                "horizon": 7,
                "metrics_json": '{"mae": 1.2, "rmse": 2.4}',
                "params_json": '{"num_samples": 10, "backend": "optuna"}',
            }
        ]
    )

    def query_df(_engine, sql: str, params=None):
        if "WHERE run_id::text = :run_id" in sql and "FROM resources.run" in sql:
            return pd.DataFrame(
                [{"run_id": "r1", "status": "success", "started_at": "2026-03-01", "ended_at": "2026-03-01", "rows_written": 100, "rows_failed": 0, "duration_sec": 10.0}]
            )
        if "FROM resources.stage_span" in sql and "GROUP BY run_id" not in sql:
            return pd.DataFrame([{"stage_name": "fit", "duration_ms": 100.0, "rows_in": 100, "rows_out": 100, "db_time_ms": 10.0, "db_rows": 1, "gpu_util_avg": 50.0, "gpu_mem_used_mb_avg": 500.0, "exception_type": None}])
        if "FROM model.nf_automodel m" in sql:
            return pd.DataFrame([{"model_name": "AutoNHITS", "run_count": 1, "avg_duration_sec": 10.0, "avg_rows_written": 100.0, "avg_rows_failed": 0.0}])
        if "FROM resources.run" in sql and "WHERE run_id::text = :run_id" not in sql:
            return pd.DataFrame([{"run_id": "r1", "run_duration_sec": 10.0, "rows_written": 100.0, "rows_failed": 0.0}])
        if "GROUP BY run_id" in sql:
            return pd.DataFrame([{"run_id": "r1", "gpu_util_avg": 50.0, "gpu_mem_avg_mb": 500.0, "stage_total_ms": 100.0}])
        return pd.DataFrame()

    shown: list[pd.DataFrame] = []
    panel.render_runid_integrated_panel(
        project_root=tmp_path,
        run_id_options=["r1"],
        run_id_to_dir={"r1": run_dir},
        model_df=model_df,
        engine=object(),
        tables={("resources", "run"), ("resources", "stage_span")},
        row_limit=20,
        settings=SimpleNamespace(default_horizon=14),
        query_df=query_df,
        show_df=lambda df, **_: shown.append(df.copy()),
        parse_json_like=lambda raw: panel.json.loads(raw) if isinstance(raw, str) else raw,
        safe_read_json_file=lambda path: {"model_name": "AutoNHITS", "h": 7, "metrics": {"mae": 1.2}},
        has_model_artifacts=lambda _: True,
        causal_proxy_ate=lambda *_args, **_kwargs: {"ok": True},
        stable_json_dumps=lambda obj: str(obj),
        plotly_available=False,
        px=None,
    )
    assert shown
    assert any(kind == "download_button" for kind, _ in fake_st.captured)
