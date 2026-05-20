import sys
import types

import pytest

from loto_forecast.data import spark_table_runner
from loto_forecast.data.spark_table_runner import (
    SparkTableRunSpec,
    _choose_execution_backend,
    _normalize_sql,
    _read_sql_for_fallback,
    _resolve_source_pushdown,
    _source_dbtable,
    _spark_mode,
)


def test_spark_mode_mapping():
    assert _spark_mode("replace") == "overwrite"
    assert _spark_mode("append") == "append"
    assert _spark_mode("fail") == "errorifexists"


def test_spark_mode_rejects_unknown():
    with pytest.raises(ValueError):
        _spark_mode("merge")


def test_source_dbtable_uses_query_if_given():
    out = _source_dbtable("dataset", "loto_y_ts", source_sql="SELECT * FROM dataset.loto_y_ts")
    assert out.startswith("(SELECT * FROM dataset.loto_y_ts)")


def test_normalize_sql_replaces_source_placeholder():
    sql = "SELECT count(*) FROM {{source}}"
    assert _normalize_sql(sql, "src_table") == "SELECT count(*) FROM src_table"


def test_read_sql_for_fallback_uses_transformed_table_ref():
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        transform_sql="SELECT * FROM {{source}} WHERE y IS NOT NULL",
    )
    out = _read_sql_for_fallback(spec)
    assert 'FROM "dataset"."loto_y_ts_unified"' in out


def test_read_sql_for_fallback_uses_source_sql_when_given():
    spec = SparkTableRunSpec(
        source_sql="SELECT * FROM dataset.loto_y_ts_unified",
        transform_sql=None,
    )
    assert _read_sql_for_fallback(spec) == "SELECT * FROM dataset.loto_y_ts_unified"


def test_resolve_source_pushdown_for_simple_transform_sql():
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        transform_sql="SELECT * FROM {{source}} WHERE y IS NOT NULL AND loto = 'bingo5'",
    )
    resolved = _resolve_source_pushdown(spec)
    assert resolved.pushdown_applied is True
    assert resolved.transform_sql is None
    assert resolved.source_sql is not None
    assert 'FROM "dataset"."loto_y_ts_unified"' in resolved.source_sql
    assert "WHERE y IS NOT NULL" in resolved.source_sql


def test_resolve_source_pushdown_keeps_complex_transform_sql():
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        transform_sql="SELECT loto, COUNT(*) AS n FROM {{source}} GROUP BY loto",
    )
    resolved = _resolve_source_pushdown(spec)
    assert resolved.pushdown_applied is False
    assert resolved.source_sql is None
    assert resolved.transform_sql == "SELECT loto, COUNT(*) AS n FROM {{source}} GROUP BY loto"


def test_choose_execution_backend_prefers_polars_in_auto(monkeypatch):
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        execution_backend="auto",
    )
    resolved = _resolve_source_pushdown(spec)

    def _fake_available(name: str) -> bool:
        return name == "polars"

    monkeypatch.setattr(spark_table_runner, "_module_available", _fake_available)
    backend, reason = _choose_execution_backend(spec, resolved)
    assert backend == "polars"
    assert reason.startswith("auto:")


def test_choose_execution_backend_uses_spark_for_non_pushdown_transform(monkeypatch):
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        execution_backend="auto",
        transform_sql="SELECT loto, COUNT(*) AS n FROM {{source}} GROUP BY loto",
    )
    resolved = _resolve_source_pushdown(spec)
    monkeypatch.setattr(spark_table_runner, "_module_available", lambda name: name == "pyspark.sql")
    backend, reason = _choose_execution_backend(spec, resolved)
    assert backend == "spark"
    assert "needs_spark_transform" in reason


def test_choose_execution_backend_respects_user_choice():
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        execution_backend="dask",
    )
    resolved = _resolve_source_pushdown(spec)
    backend, reason = _choose_execution_backend(spec, resolved)
    assert backend == "dask"
    assert reason == "user:dask"


def test_run_table_with_pyspark_fallback_on_runtime_error(monkeypatch):
    fake_pyspark = types.ModuleType("pyspark")
    fake_sql = types.ModuleType("pyspark.sql")
    fake_sql.SparkSession = object
    monkeypatch.setitem(sys.modules, "pyspark", fake_pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_sql)

    class _FakeReader:
        def format(self, *_args, **_kwargs):
            return self

        def option(self, *_args, **_kwargs):
            return self

        def load(self):
            raise RuntimeError("ClassNotFoundException: org.postgresql.Driver")

    class _FakeSpark:
        def __init__(self):
            self.read = _FakeReader()
            self.stopped = False

        def stop(self):
            self.stopped = True

    fake_spark = _FakeSpark()
    monkeypatch.setattr(spark_table_runner, "_create_spark_session", lambda spec: (fake_spark, {"strategy": "test"}))
    monkeypatch.setattr(
        spark_table_runner,
        "_run_with_pandas_fallback",
        lambda spec, spark_error: {"ok": True, "fallback_engine": "pandas", "fallback_reason": spark_error},
    )

    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        execution_backend="spark",
        fallback_to_pandas=True,
    )
    out = spark_table_runner.run_table_with_pyspark(spec)
    assert out["ok"] is True
    assert out["fallback_engine"] == "pandas"
    assert "org.postgresql.Driver" in out["fallback_reason"]
    assert fake_spark.stopped is True


def test_run_table_with_pyspark_prefer_pandas_short_circuit(monkeypatch):
    called = {"n": 0}

    def _fake_fallback(spec, spark_error):
        called["n"] += 1
        return {"ok": True, "fallback_engine": "pandas", "fallback_reason": spark_error}

    monkeypatch.setattr(spark_table_runner, "_run_with_pandas_fallback", _fake_fallback)
    spec = SparkTableRunSpec(
        source_schema="dataset",
        source_table="loto_y_ts_unified",
        prefer_pandas=True,
    )
    out = spark_table_runner.run_table_with_pyspark(spec)
    assert out["ok"] is True
    assert out["fallback_engine"] == "pandas"
    assert "prefer_pandas=true" in out["fallback_reason"]
    assert called["n"] == 1
