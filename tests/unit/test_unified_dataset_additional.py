from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from loto_forecast.data import unified_dataset as mod


class _FakeExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((str(query), params))
        return _FakeExecuteResult(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self, connect_rows=None):
        self.connect_rows = connect_rows or []
        self.connect_calls = []
        self.begin_conn = _FakeConn()

    def connect(self):
        conn = _FakeConn(self.connect_rows)
        self.connect_calls.append(conn)
        return conn

    def begin(self):
        return self.begin_conn


def test_identifier_helpers_and_progress() -> None:
    assert mod._safe_ident('a"b-c_1') == "abc_1"
    assert mod._quote_ident('a"b') == '"a""b"'
    assert mod._table_ref("schema-x", "table y") == '"schemax"."tabley"'
    assert mod._noop_progress("x") is None
    progress = mod._make_progress(False)
    assert progress("ignored") is None

    with pytest.raises(ValueError):
        mod._safe_ident("!!!")


def test_list_exists_read_and_select_exog_tables(monkeypatch) -> None:
    engine = _FakeEngine(connect_rows=[("hist_a",), ("hist_b",)])
    assert mod._list_tables(engine, "exog") == ["hist_a", "hist_b"]

    engine_exists = _FakeEngine(connect_rows=[(1,)])
    assert mod._table_exists(engine_exists, "dataset", "tbl") is True

    captured = {}

    def _fake_read_sql(sql, _engine):
        captured["sql"] = sql
        return pd.DataFrame({"ds": ["2025-01-01"], "y": [1.0]})

    monkeypatch.setattr(pd, "read_sql", _fake_read_sql)
    out = mod._read_table(object(), "dataset", "tbl", "ds")
    assert "SELECT * FROM" in captured["sql"]
    assert str(out.loc[0, "ds"].date()) == "2025-01-01"

    spec = mod.UnifiedBuildSpec(include_exog_tables=("a", "b"), exclude_exog_tables=("b",))
    assert mod._select_exog_tables(spec, ["a", "b", "c"]) == ["a"]


def test_build_unified_dataset_joins_history_and_exog(monkeypatch) -> None:
    base = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "y": [1.0, 2.0],
        }
    )
    hist = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2025-01-02"]),
            "hist_signal": [5.0],
        }
    )
    exog = pd.DataFrame(
        {
            "unique_id": ["A"],
            "ds": pd.to_datetime(["2025-01-02"]),
            "hist_signal": [7.0],
        }
    )

    monkeypatch.setattr(mod, "_table_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mod, "_list_tables", lambda *_args, **_kwargs: ["weather"])

    def _fake_read_table(_engine, schema, table, _time_col):
        if schema == "dataset" and table == "loto_y_ts":
            return base.copy()
        if schema == "dataset" and table == "loto_hist_feat":
            return hist.copy()
        if schema == "exog" and table == "weather":
            return exog.copy()
        raise AssertionError((schema, table))

    monkeypatch.setattr(mod, "_read_table", _fake_read_table)

    res = mod.build_unified_dataset(object(), mod.UnifiedBuildSpec(), progress=lambda _msg: None)

    assert res.joined_tables == ["dataset.loto_hist_feat", "exog.weather"]
    assert res.join_keys_by_table["dataset.loto_hist_feat"] == ["unique_id", "ds"]
    assert "hist_weather_signal" in res.dataframe.columns
    assert float(res.dataframe.loc[1, "hist_signal"]) == 5.0
    assert float(res.dataframe.loc[1, "hist_weather_signal"]) == 7.0


def test_build_unified_dataset_records_skipped_tables(monkeypatch) -> None:
    base = pd.DataFrame({"unique_id": ["A"], "ds": pd.to_datetime(["2025-01-01"]), "y": [1.0]})
    exog = pd.DataFrame({"other_key": ["x"], "value": [2.0]})

    monkeypatch.setattr(mod, "_table_exists", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mod, "_list_tables", lambda *_args, **_kwargs: ["weather"])

    def _fake_read_table(_engine, schema, table, _time_col):
        if schema == "dataset":
            return base.copy()
        return exog.copy()

    monkeypatch.setattr(mod, "_read_table", _fake_read_table)

    res = mod.build_unified_dataset(object(), mod.UnifiedBuildSpec(), progress=lambda _msg: None)

    assert {"table": "dataset.loto_hist_feat", "reason": "table not found"} in res.skipped_tables
    assert {"table": "exog.weather", "reason": "no common join keys"} in res.skipped_tables


def test_write_to_postgres_uses_to_sql_path_and_creates_index(monkeypatch) -> None:
    calls = {"to_sql": []}
    engine = _FakeEngine()
    df = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})

    def _fake_to_sql(self, table, engine_arg, **kwargs):
        calls["to_sql"].append((table, engine_arg, kwargs))

    monkeypatch.setattr(pd.DataFrame, "to_sql", _fake_to_sql)

    spec = mod.UnifiedBuildSpec(output_table="out_tbl", output_if_exists="append", postgres_write_mode="to_sql")
    out = mod._write_to_postgres(engine, df, spec, progress=lambda _msg: None)

    assert out == {"schema": "dataset", "table": "out_tbl", "rows": 1}
    assert calls["to_sql"][0][0] == "out_tbl"
    assert any("CREATE SCHEMA IF NOT EXISTS dataset" in sql for sql, _ in engine.begin_conn.executed)
    assert any("CREATE INDEX IF NOT EXISTS" in sql for sql, _ in engine.begin_conn.executed)


def test_write_helpers_and_persist_outputs(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})
    csv_path = tmp_path / "out.csv"
    parquet_path = tmp_path / "out.parquet"

    csv_out = mod._write_csv(df, str(csv_path))
    parquet_out = mod._write_parquet(df, str(parquet_path))
    spark_out = mod._write_spark(df, str(tmp_path / "spark"), "parquet")

    assert csv_out["rows"] == 1 and csv_path.exists()
    assert parquet_out["rows"] == 1 and parquet_path.exists()
    assert spark_out["spark_available"] is False

    captured = []
    monkeypatch.setattr(mod, "_write_to_postgres", lambda *_args, **_kwargs: {"rows": 1, "table": "x"})
    monkeypatch.setattr(mod, "_write_csv", lambda *_args, **_kwargs: captured.append("csv") or {"rows": 1})
    monkeypatch.setattr(mod, "_write_parquet", lambda *_args, **_kwargs: captured.append("parquet") or {"rows": 1})
    monkeypatch.setattr(mod, "_write_spark", lambda *_args, **_kwargs: captured.append("spark") or {"rows": 1})

    spec = mod.UnifiedBuildSpec(
        output_csv_path="a.csv",
        output_parquet_path="a.parquet",
        output_spark_path="spark_out",
    )
    outputs = mod.persist_unified_outputs(object(), df, spec, progress=lambda _msg: None)

    assert outputs["postgres"]["rows"] == 1
    assert captured == ["csv", "parquet", "spark"]


@dataclass
class _ReadSqlCase:
    columns_df: pd.DataFrame
    count_df: pd.DataFrame
    null_df: pd.DataFrame
    dup_group_df: pd.DataFrame
    dup_group_time_df: pd.DataFrame
    sample_df: pd.DataFrame


def test_check_unified_grouping_in_table_handles_missing_columns(monkeypatch) -> None:
    case = _ReadSqlCase(
        columns_df=pd.DataFrame({"column_name": ["unique_id", "ds"]}),
        count_df=pd.DataFrame({"n": [1]}),
        null_df=pd.DataFrame({"n": [0]}),
        dup_group_df=pd.DataFrame({"n": [0]}),
        dup_group_time_df=pd.DataFrame({"n": [0]}),
        sample_df=pd.DataFrame(),
    )
    engine = _FakeEngine()

    monkeypatch.setattr(mod, "create_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        pd,
        "read_sql",
        lambda query, _conn, params=None: case.columns_df if "information_schema.columns" in str(query) else case.count_df,
    )

    out = mod.check_unified_grouping_in_table("h", 5432, "u", "p", "db", group_cols=("unique_id", "ts_type"))

    assert out["ok"] is False
    assert out["missing_columns"] == ["ts_type"]


def test_check_unified_grouping_in_table_returns_summary(monkeypatch) -> None:
    frames = [
        pd.DataFrame({"column_name": ["unique_id", "ts_type", "ds"]}),
        pd.DataFrame({"n": [10]}),
        pd.DataFrame({"n": [0]}),
        pd.DataFrame({"n": [1]}),
        pd.DataFrame({"n": [0]}),
        pd.DataFrame({"unique_id": ["A"], "ts_type": ["raw"], "n_rows": [10], "min_time": ["2025-01-01"], "max_time": ["2025-01-10"]}),
    ]
    engine = _FakeEngine()

    monkeypatch.setattr(mod, "create_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(pd, "read_sql", lambda *args, **kwargs: frames.pop(0))

    out = mod.check_unified_grouping_in_table("h", 5432, "u", "p", "db", group_cols=("unique_id", "ts_type"))

    assert out["ok"] is True
    assert out["row_count"] == 10
    assert out["duplicate_group_rows"] == 1
    assert out["sample_groups"][0]["unique_id"] == "A"
