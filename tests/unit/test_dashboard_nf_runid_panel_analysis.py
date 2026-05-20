from __future__ import annotations

import numpy as np
import pandas as pd

from loto_forecast.api.streamlit import dashboard_nf_runid_panel_analysis as analysis


def test_merge_resource_features_and_feature_columns() -> None:
    base = pd.DataFrame({"run_id": ["r1"], "metric.mae": [1.0], "param.depth": [3.0], "horizon": [7]})
    run_feat = pd.DataFrame({"run_id": ["r1"], "run_duration_sec": [10.0]})
    stage_feat = pd.DataFrame({"run_id": ["r1"], "gpu_util_avg": [40.0]})
    merged = analysis.merge_resource_features(base, run_feat=run_feat, stage_feat=stage_feat)
    assert "run_duration_sec" in merged.columns
    assert "gpu_util_avg" in merged.columns
    assert analysis.feature_columns(merged) == ["param.depth", "horizon", "run_duration_sec", "gpu_util_avg"]


def test_build_correlation_rows() -> None:
    df = pd.DataFrame(
        {
            "metric.mae": np.arange(20, dtype=float),
            "param.depth": np.arange(20, dtype=float),
            "run_duration_sec": np.arange(20, dtype=float) * 2,
        }
    )
    corr = analysis.build_correlation_rows(df, target_col="metric.mae")
    assert set(corr["feature"]) == {"param.depth", "run_duration_sec"}
