import pandas as pd

from loto_forecast.data.unified_dataset import (
    UnifiedBuildSpec,
    _deduplicate_for_join,
    _effective_spec,
    _infer_join_keys,
    _prefix_for_collision,
    _write_to_postgres_copy,
)


def test_infer_join_keys_prefers_row_id():
    spec = UnifiedBuildSpec()
    keys = _infer_join_keys(
        left_cols=["row_id", "unique_id", "ds", "y"],
        right_cols=["row_id", "hist_x"],
        spec=spec,
    )
    assert keys == ["row_id"]


def test_infer_join_keys_prefers_unique_id_and_ds():
    spec = UnifiedBuildSpec(id_col="unique_id", time_col="ds")
    keys = _infer_join_keys(
        left_cols=["unique_id", "ds", "y"],
        right_cols=["unique_id", "ds", "hist_x"],
        spec=spec,
    )
    assert keys[:2] == ["unique_id", "ds"]


def test_prefix_for_collision_preserves_exog_role():
    assert _prefix_for_collision("timesfm", "hist_signal") == "hist_timesfm_signal"
    assert _prefix_for_collision("timesfm", "stat_signal") == "stat_timesfm_signal"
    assert _prefix_for_collision("timesfm", "feat_signal") == "feat_timesfm_signal"


def test_deduplicate_for_join_keeps_last():
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "hist_x": [1.0, 2.0],
        }
    )
    out = _deduplicate_for_join(df, ["unique_id", "ds"], "ds")
    assert len(out) == 1
    assert float(out.iloc[0]["hist_x"]) == 2.0


def test_effective_spec_fast_mode_enables_copy_and_disables_heavy_outputs():
    spec = UnifiedBuildSpec(
        fast_mode=True,
        output_csv_path="./x.csv",
        output_parquet_path="./x.parquet",
        output_spark_path="./x_spark",
        sort_output=True,
        create_postgres_index=True,
        postgres_write_mode="to_sql",
        postgres_copy_chunk_rows=5000,
        postgres_chunksize=5000,
    )
    out = _effective_spec(spec)
    assert out.output_csv_path is None
    assert out.output_parquet_path is None
    assert out.output_spark_path is None
    assert out.sort_output is False
    assert out.create_postgres_index is False
    assert out.postgres_write_mode == "copy"
    assert out.postgres_copy_chunk_rows >= 20000


def test_write_to_postgres_copy_uses_single_backslash_null_marker(monkeypatch):
    class _FakeCursor:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def copy_expert(self, sql, buf):
            self.calls.append((sql, buf.getvalue()))

    class _FakeRawConnection:
        def __init__(self):
            self.cursor_obj = _FakeCursor()
            self.committed = False
            self.rolled_back = False
            self.closed = False

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    class _FakeEngine:
        def __init__(self):
            self.raw = _FakeRawConnection()

        def raw_connection(self):
            return self.raw

    monkeypatch.setattr(pd.DataFrame, "to_sql", lambda self, *args, **kwargs: None)

    engine = _FakeEngine()
    df = pd.DataFrame({"hist_pn5": [1.5, None], "unique_id": ["A", "B"]})
    spec = UnifiedBuildSpec(postgres_copy_chunk_rows=1000)

    _write_to_postgres_copy(
        engine=engine,
        df=df,
        schema="dataset",
        table="loto_y_ts_unified",
        spec=spec,
        progress=lambda _: None,
    )

    assert len(engine.raw.cursor_obj.calls) == 1
    copy_sql, payload = engine.raw.cursor_obj.calls[0]
    assert "NULL '\\N'" in copy_sql
    assert "\\N,B" in payload
    assert engine.raw.committed is True
    assert engine.raw.rolled_back is False
    assert engine.raw.closed is True
