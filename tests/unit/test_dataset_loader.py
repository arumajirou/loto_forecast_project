from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from loto_forecast.data import dataset_loader as mod


def test_basic_normalizers() -> None:
    assert mod._to_bool(True) is True
    assert mod._to_bool("yes") is True
    assert mod._to_bool("0", default=True) is False
    assert mod._to_bool("unknown", default=True) is True

    assert mod._to_json_dict({"a": 1}) == {"a": 1}
    assert mod._to_json_dict('{"a": 1}') == {"a": 1}
    assert mod._to_json_dict("[1,2]") == {}
    assert mod._to_json_dict(None) == {}

    assert mod._normalize_loader_name("postgres") == "db_table"
    assert mod._normalize_loader_name("ndjson") == "jsonl"
    assert mod._normalize_loader_name("query") == "sql"
    assert mod._normalize_backend_name("spark") == "spark"
    assert mod._normalize_backend_name("unknown") == "pandas"


def test_resolve_input_paths_supports_file_dir_and_glob(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "nested" / "b.csv"
    csv_b.parent.mkdir()
    csv_a.write_text("x\n1\n", encoding="utf-8")
    csv_b.write_text("x\n2\n", encoding="utf-8")

    single = mod._resolve_input_paths("csv", str(csv_a), load_all=True)
    dir_all = mod._resolve_input_paths("csv", str(tmp_path), load_all=True)
    glob_one = mod._resolve_input_paths("csv", str(tmp_path / "**/*.csv"), load_all=False)

    assert single == [csv_a.resolve()]
    assert dir_all == [csv_a.resolve(), csv_b.resolve()]
    assert glob_one == [csv_a.resolve()]


def test_read_one_with_pandas_variants(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    jsonl_path = tmp_path / "sample.jsonl"
    csv_path.write_text("unique_id,ds,y\nA,2025-01-01,1.0\n", encoding="utf-8")
    jsonl_path.write_text('{"unique_id":"A","ds":"2025-01-01","y":1.0}\n', encoding="utf-8")

    csv_df = mod._read_one_with_pandas(csv_path, "csv", {})
    jsonl_df = mod._read_one_with_pandas(jsonl_path, "jsonl", {})

    assert csv_df["unique_id"].tolist() == ["A"]
    assert jsonl_df["y"].tolist() == [1.0]

    with pytest.raises(ValueError):
        mod._read_one_with_pandas(csv_path, "xml", {})


def test_read_file_dataset_uses_backends_and_fallbacks(tmp_path: Path, monkeypatch) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("unique_id,ds,y\nA,2025-01-01,1.0\n", encoding="utf-8")
    csv_b.write_text("unique_id,ds,y\nB,2025-01-02,2.0\n", encoding="utf-8")

    class _FakePolarsFrame:
        def to_pandas(self):
            return pd.DataFrame({"unique_id": ["P"], "ds": ["2025-01-03"], "y": [3.0]})

    class _FakeDaskFrame:
        def compute(self):
            return pd.DataFrame({"unique_id": ["D"], "ds": ["2025-01-04"], "y": [4.0]})

    fake_polars = types.SimpleNamespace(read_csv=lambda *_args, **_kwargs: _FakePolarsFrame())
    fake_dask_df = types.SimpleNamespace(read_csv=lambda *_args, **_kwargs: _FakeDaskFrame())
    fake_dask_pkg = types.SimpleNamespace(dataframe=fake_dask_df)

    monkeypatch.setitem(sys.modules, "polars", fake_polars)
    monkeypatch.setitem(sys.modules, "dask", fake_dask_pkg)
    monkeypatch.setitem(sys.modules, "dask.dataframe", fake_dask_df)

    out_polars, meta_polars = mod._read_file_dataset("csv", "polars", str(tmp_path), load_all=False)
    out_dask, meta_dask = mod._read_file_dataset("csv", "dask", str(tmp_path), load_all=False)
    out_fallback, meta_fallback = mod._read_file_dataset("csv", "spark", str(tmp_path), load_all=True)

    assert out_polars["unique_id"].tolist() == ["P"]
    assert meta_polars["backend_used"] == "polars"
    assert out_dask["unique_id"].tolist() == ["D"]
    assert meta_dask["backend_used"] == "dask"
    assert sorted(out_fallback["unique_id"].tolist()) == ["A", "B"]
    assert meta_fallback["backend_used"] == "pandas"
    assert meta_fallback["source_count"] == 2
    assert "backend=spark not enabled" in meta_fallback["warnings"][0]


def test_read_file_dataset_raises_for_missing_and_reader_errors(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(FileNotFoundError):
        mod._read_file_dataset("csv", "pandas", str(tmp_path / "missing"), load_all=False)

    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(mod, "_read_one_with_pandas", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")))

    with pytest.raises(RuntimeError, match="failed to read dataset file"):
        mod._read_file_dataset("csv", "pandas", str(csv_path), load_all=False)


def test_load_dataset_from_settings_db_sql_and_files(monkeypatch, tmp_path: Path) -> None:
    df = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})
    monkeypatch.setattr(mod, "read_timeseries", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(pd, "read_sql", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(mod, "_read_file_dataset", lambda **_kwargs: (df.copy(), {"source_count": 1, "sources": ["f"]}))

    db_df, db_meta = mod.load_dataset_from_settings(
        object(),
        {"dataset_loader": "db_table", "dataset_schema": "dataset", "dataset_table": "tbl"},
        default_schema="default",
        default_table="base",
    )
    sql_df, sql_meta = mod.load_dataset_from_settings(
        object(),
        {"dataset_loader": "sql", "dataset_sql": "SELECT 1", "dataset_sql_params": '{"a": 1}'},
        default_schema="default",
        default_table="base",
    )
    file_df, file_meta = mod.load_dataset_from_settings(
        object(),
        {"dataset_loader": "csv", "dataset_path": str(tmp_path), "dataset_load_all": False},
        default_schema="default",
        default_table="base",
    )

    assert db_df.equals(df)
    assert db_meta["label"] == "dataset.tbl"
    assert sql_df.equals(df)
    assert sql_meta["label"] == "sql_query"
    assert file_df.equals(df)
    assert file_meta["source_count"] == 1

    with pytest.raises(ValueError, match="dataset_loader=sql requires dataset_sql"):
        mod.load_dataset_from_settings(object(), {"dataset_loader": "sql"}, default_schema="d", default_table="t")
