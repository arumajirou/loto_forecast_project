import numpy as np
import pandas as pd

from resources.uni2ts_exog_pipeline import Uni2TSExogSpec, build_uni2ts_exog_dataframe


def _sample_df(include_row_id: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "loto": ["A"] * 5 + ["B"] * 5,
            "unique_id": ["U1"] * 5 + ["U2"] * 5,
            "ts_type": ["daily"] * 10,
            "ds": pd.date_range("2025-01-01", periods=5, freq="D").tolist()
            + pd.date_range("2025-01-01", periods=5, freq="D").tolist(),
            "y": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14],
        }
    )
    if include_row_id:
        df.insert(0, "loto_y_ts_row_id", np.arange(100, 100 + len(df), dtype=np.int64))
    return df


def test_build_uni2ts_exog_shape_and_columns() -> None:
    spec = Uni2TSExogSpec(embedding_dim=8, context_length=4, parallel_workers=1, enable_gpu_compute=False)
    out, meta = build_uni2ts_exog_dataframe(_sample_df(), spec)

    assert len(out) == 10
    assert "loto_y_ts_row_id" in out.columns
    for i in range(1, 9):
        assert f"hist_uni2ts_{i}" in out.columns
    assert all(c.startswith("hist_uni2ts_") for c in out.columns if c.startswith("hist_uni2ts_"))

    assert "embedding_dim" in out.columns
    assert "model_name" in out.columns
    assert "model_version" in out.columns
    assert "config_hash" in out.columns
    assert "created_at" in out.columns
    assert "updated_at" in out.columns
    assert "y_idx" in out.columns

    assert meta["embedding_dim"] == 8
    assert meta["context_length"] == 4


def test_preserve_source_row_id_if_present() -> None:
    spec = Uni2TSExogSpec(embedding_dim=4, context_length=4, parallel_workers=1, enable_gpu_compute=False)
    src = _sample_df(include_row_id=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    out, _ = build_uni2ts_exog_dataframe(src, spec)

    expected = src.sort_values(["loto", "unique_id", "ts_type", "ds"])["loto_y_ts_row_id"].tolist()
    actual = out["loto_y_ts_row_id"].tolist()
    assert actual == expected


def test_fallback_embedding_is_deterministic() -> None:
    spec = Uni2TSExogSpec(embedding_dim=16, context_length=8, parallel_workers=1, enable_gpu_compute=False)
    src = _sample_df(include_row_id=False)

    out1, _ = build_uni2ts_exog_dataframe(src, spec)
    out2, _ = build_uni2ts_exog_dataframe(src, spec)

    cols = [c for c in out1.columns if c.startswith("hist_uni2ts_")]
    assert cols
    np.testing.assert_allclose(
        out1[cols].to_numpy(dtype=np.float32), out2[cols].to_numpy(dtype=np.float32), rtol=1e-6, atol=1e-6
    )
