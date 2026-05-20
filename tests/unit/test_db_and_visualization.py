from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from loto_forecast.analysis import visualization
from loto_forecast.data import db as dbmod


class _FakeResult:
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
        self.driver_sql = []

    def execute(self, query, params=None):
        self.executed.append((str(query), params))
        return _FakeResult(self.rows)

    def exec_driver_sql(self, sql):
        self.driver_sql.append(sql)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self, connect_rows=None):
        self.connect_rows = connect_rows or []
        self.begin_conn = _FakeConn()

    def connect(self):
        return _FakeConn(self.connect_rows)

    def begin(self):
        return self.begin_conn


def test_db_helpers(monkeypatch, tmp_path: Path) -> None:
    engine = _FakeEngine(connect_rows=[("unique_id", "text"), ("ds", "timestamp")])
    cols = dbmod.table_columns(engine, "dataset", "tbl")
    assert cols == [("unique_id", "text"), ("ds", "timestamp")]

    exists_engine = _FakeEngine(connect_rows=[(1,)])
    assert dbmod.table_exists(exists_engine, "dataset", "tbl") is True

    captured = {}

    def _fake_read_sql(sql, _engine, params=None):
        captured["sql"] = sql
        captured["params"] = params
        return pd.DataFrame({"ds": ["2025-01-01"], "y": [1.0]})

    monkeypatch.setattr(pd, "read_sql", _fake_read_sql)
    ts_df = dbmod.read_timeseries(object(), "dataset", "tbl", where_sql="y > 0")
    qry_df = dbmod.read_query(object(), "SELECT * FROM x", params={"a": 1})

    sql_text = str(captured["sql"])
    assert "WHERE y > 0" in sql_text or sql_text.startswith("SELECT * FROM")
    assert str(ts_df.loc[0, "ds"].date()) == "2025-01-01"
    assert qry_df.loc[0, "y"] == 1.0

    dbmod.execute_sql(engine, "SELECT 1")
    assert engine.begin_conn.driver_sql == ["SELECT 1"]

    sql_file = tmp_path / "query.sql"
    sql_file.write_text("SELECT 2", encoding="utf-8")
    dbmod.execute_sql_file(engine, str(sql_file))
    assert engine.begin_conn.driver_sql[-1] == "SELECT 2"


def test_make_engine_and_visualization_helpers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dbmod, "create_engine", lambda url, pool_pre_ping=True: {"url": url, "pool_pre_ping": pool_pre_ping})
    engine = dbmod.make_engine()
    assert engine["pool_pre_ping"] is True
    assert "postgresql+psycopg2://" in engine["url"]

    actual = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "ds": pd.date_range("2025-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "ds": pd.date_range("2025-01-01", periods=3, freq="D"),
            "Model": [1.1, 1.9, 2.5],
        }
    )
    out_plot = visualization.plot_forecast_vs_actual(actual, forecast, "Model", tmp_path / "plots" / "fcst.png", id_value="A")
    assert out_plot.exists()

    imp = pd.DataFrame({"feature": ["x1", "x2"], "delta_mae_mean": [0.5, 0.2]})
    out_imp = visualization.plot_exog_importance(imp, tmp_path / "plots" / "importance.png")
    assert out_imp.exists()

    with pytest.raises(ValueError, match="importance_df is empty"):
        visualization.plot_exog_importance(pd.DataFrame(), tmp_path / "plots" / "empty.png")
