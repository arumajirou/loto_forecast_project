import numpy as np
import pandas as pd

from resources.chronos_exog_pipeline import ChronosExogSpec, build_chronos_exog_dataframe


def _sample_df(with_row_id: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "loto": ["A"] * 6 + ["B"] * 6,
            "unique_id": ["u1"] * 6 + ["u2"] * 6,
            "ts_type": ["daily"] * 12,
            "ds": pd.date_range("2025-01-01", periods=6, freq="D").tolist()
            + pd.date_range("2025-01-01", periods=6, freq="D").tolist(),
            "y": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15],
        }
    )
    if with_row_id:
        df.insert(0, "row_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


def test_build_chronos_exog_columns() -> None:
    spec = ChronosExogSpec(
        backend="chronos_forecast_features",
        embedding_dim=8,
        window_size=4,
        min_points=2,
        parallel_workers=2,
        enable_gpu_compute=False,
    )
    out, meta = build_chronos_exog_dataframe(_sample_df(with_row_id=True), spec, target_ids=None)

    assert "loto_y_ts_row_id" in out.columns
    assert "y_idx" in out.columns
    for i in range(1, 9):
        assert f"hist_chronos_{i}" in out.columns
    assert "embedding_dim" in out.columns
    assert "model_name" in out.columns
    assert "model_version" in out.columns
    assert "config_hash" in out.columns
    assert "created_at" in out.columns
    assert "updated_at" in out.columns

    assert meta["resolved_backend"] == "chronos_forecast_features"
    assert meta["embedding_dim"] == 8


def test_build_chronos_exog_target_id_filter() -> None:
    spec = ChronosExogSpec(
        backend="chronos_forecast_features",
        embedding_dim=4,
        window_size=4,
        min_points=1,
        parallel_workers=1,
        enable_gpu_compute=False,
    )
    src = _sample_df(with_row_id=True)
    target_ids = {2, 4, 7}
    out, _ = build_chronos_exog_dataframe(src, spec, target_ids=target_ids)
    assert not out.empty
    assert set(out["loto_y_ts_row_id"].tolist()).issubset(target_ids)


def test_build_chronos_exog_deterministic() -> None:
    spec = ChronosExogSpec(
        backend="chronos_forecast_features",
        embedding_dim=16,
        window_size=5,
        min_points=2,
        parallel_workers=1,
        enable_gpu_compute=False,
    )
    src = _sample_df(with_row_id=False)
    out1, _ = build_chronos_exog_dataframe(src, spec, target_ids=None)
    out2, _ = build_chronos_exog_dataframe(src, spec, target_ids=None)

    cols = [c for c in out1.columns if c.startswith("hist_chronos_")]
    assert cols
    np.testing.assert_allclose(
        out1[cols].to_numpy(dtype=np.float32),
        out2[cols].to_numpy(dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
