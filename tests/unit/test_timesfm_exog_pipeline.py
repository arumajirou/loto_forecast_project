import numpy as np
import pandas as pd

from resources.timesfm_exog_pipeline import TimesFMExogSpec, build_timesfm_exog_dataframe


def _sample_df(with_row_id: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "loto": ["A"] * 5 + ["B"] * 5,
            "ts_type": ["daily"] * 10,
            "ds": pd.date_range("2025-01-01", periods=5, freq="D").tolist()
            + pd.date_range("2025-01-01", periods=5, freq="D").tolist(),
            "y": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14],
            "unique_id": ["u1"] * 5 + ["u2"] * 5,
        }
    )
    if with_row_id:
        df.insert(0, "row_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


def test_build_timesfm_exog_columns() -> None:
    spec = TimesFMExogSpec(
        embedding_dim=8,
        window_size=4,
        min_points=2,
        backend="timesfm_forecast_features",
        parallel_workers=1,
        enable_gpu_compute=False,
        group_cols=("loto", "ts_type"),
    )
    out, meta = build_timesfm_exog_dataframe(_sample_df(with_row_id=True), spec, target_ids=None)

    assert "loto_y_ts_row_id" in out.columns
    assert "y_idx" in out.columns
    for i in range(1, 9):
        assert f"hist_timesfm_{i}" in out.columns
    assert all(c.startswith("hist_timesfm_") for c in out.columns if c.startswith("hist_timesfm_"))
    assert "embedding_dim" in out.columns
    assert "model_name" in out.columns
    assert "model_version" in out.columns
    assert "config_hash" in out.columns
    assert "created_at" in out.columns
    assert "updated_at" in out.columns
    assert meta["resolved_backend"] == "timesfm_forecast_features"
    assert meta["embedding_dim"] == 8


def test_build_timesfm_exog_target_id_filter() -> None:
    spec = TimesFMExogSpec(
        embedding_dim=4,
        window_size=4,
        min_points=1,
        backend="timesfm_forecast_features",
        parallel_workers=1,
        enable_gpu_compute=False,
        group_cols=("loto", "ts_type"),
    )
    src = _sample_df(with_row_id=True)
    target_ids = {2, 4, 7}
    out, _ = build_timesfm_exog_dataframe(src, spec, target_ids=target_ids)

    assert not out.empty
    assert set(out["loto_y_ts_row_id"].tolist()).issubset(target_ids)


def test_build_timesfm_exog_deterministic() -> None:
    spec = TimesFMExogSpec(
        embedding_dim=16,
        window_size=5,
        min_points=2,
        backend="timesfm_forecast_features",
        parallel_workers=1,
        enable_gpu_compute=False,
        group_cols=("loto", "ts_type"),
    )
    src = _sample_df(with_row_id=False)
    out1, _ = build_timesfm_exog_dataframe(src, spec, target_ids=None)
    out2, _ = build_timesfm_exog_dataframe(src, spec, target_ids=None)

    cols = [c for c in out1.columns if c.startswith("hist_timesfm_")]
    assert cols
    np.testing.assert_allclose(
        out1[cols].to_numpy(dtype=np.float32), out2[cols].to_numpy(dtype=np.float32), rtol=1e-6, atol=1e-6
    )
