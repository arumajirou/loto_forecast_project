from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_runid_panel_formatter as formatter


def test_overview_resource_accuracy_and_export_formatters(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "forecast.parquet").write_text("x", encoding="utf-8")
    metrics = formatter.build_overview_metrics(
        sel_run="r1",
        sel_meta={"model_name": "Auto", "h": 7},
        sel_model_row={"status": "success"},
        sel_dir=run_dir,
        sel_eval={"mae": 1.2},
        settings=SimpleNamespace(default_horizon=14),
        has_model_artifacts=lambda _: True,
    )
    assert metrics[0]["value"] == "r1"
    assert metrics[4]["value"] == "yes"

    resource_metrics = formatter.build_resource_metrics(pd.Series({"duration_sec": 10.0, "rows_written": 5, "rows_failed": 1, "status": "done"}))
    assert resource_metrics[0]["value"] == "10.00"

    use_df = pd.DataFrame({"model_name": ["A", "A", "B"], "value": [1.0, 2.0, 3.0]})
    agg = formatter.build_accuracy_aggregate(use_df)
    assert agg.iloc[0]["model_name"] == "A"

    payload = formatter.build_export_payload(
        sel_run="r1",
        sel_meta={"a": 1},
        sel_eval={"b": 2},
        sel_model_row={"c": 3},
        sel_params={"d": 4},
        sel_metrics={"e": 5},
    )
    assert payload["run_id"] == "r1"
    assert "run_id" in formatter.build_export_preview(payload, stable_json_dumps=lambda obj: str(obj))
