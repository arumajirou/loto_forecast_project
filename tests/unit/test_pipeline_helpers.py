from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from loto_forecast.orchestration import pipeline


def test_to_json_dict_runtime_kwargs_and_normalizers() -> None:
    assert pipeline._to_json_dict({"a": 1}) == {"a": 1}
    assert pipeline._to_json_dict('{"a": 1}') == {"a": 1}
    assert pipeline._to_json_dict("{bad json") == {}
    assert pipeline._to_json_dict(123) == {}

    kwargs = pipeline._read_nf_runtime_kwargs({"nf_runtime_kwargs": {"nf_fit_kwargs": {"epochs": 3}}})
    assert kwargs["nf_fit_kwargs"] == {"epochs": 3}
    assert kwargs["nf_predict_kwargs"] == {}

    assert pipeline._normalize_dataset_input_method("CSV") == "csv"
    assert pipeline._normalize_dataset_input_method("unknown", default="db_sql") == "db_sql"
    assert pipeline._normalize_dataframe_backend("SPARK") == "spark"
    assert pipeline._normalize_dataframe_backend("unknown", default="pandas") == "pandas"


def test_load_dataset_from_source_db_variants(monkeypatch) -> None:
    df = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})

    monkeypatch.setattr(pipeline, "read_timeseries", lambda *_args, **_kwargs: df.copy())
    out_df, source = pipeline._load_dataset_from_source(
        object(),
        input_method="db_table",
        dataframe_backend="pandas",
        dataset_schema="dataset",
        dataset_table="tbl",
        dataset_where="y > 0",
        dataset_sql=None,
        dataset_path=None,
    )
    assert source == "dataset.tbl"
    assert out_df.equals(df)

    class _FakeBegin:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: df.copy())
    sql_df, sql_source = pipeline._load_dataset_from_source(
        types.SimpleNamespace(begin=lambda: _FakeBegin()),
        input_method="db_sql",
        dataframe_backend="pandas",
        dataset_schema="dataset",
        dataset_table="tbl",
        dataset_where=None,
        dataset_sql="SELECT * FROM x",
        dataset_path=None,
    )
    assert sql_source == "db_sql"
    assert sql_df.equals(df)


def test_load_dataset_from_source_file_variants_and_errors(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    json_path = tmp_path / "sample.json"
    csv_df = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})
    csv_df.to_csv(csv_path, index=False)
    json_path.write_text('{"unique_id": "A", "ds": "2025-01-01", "y": 1.0}\n', encoding="utf-8")

    out_csv, csv_source = pipeline._load_dataset_from_source(
        object(),
        input_method="csv",
        dataframe_backend="pandas",
        dataset_schema="dataset",
        dataset_table="tbl",
        dataset_where=None,
        dataset_sql=None,
        dataset_path=str(csv_path),
    )
    out_json, json_source = pipeline._load_dataset_from_source(
        object(),
        input_method="json",
        dataframe_backend="pandas",
        dataset_schema="dataset",
        dataset_table="tbl",
        dataset_where=None,
        dataset_sql=None,
        dataset_path=str(json_path),
    )

    assert csv_source.startswith("csv:")
    assert json_source.startswith("json:")
    assert str(out_csv.loc[0, "ds"].date()) == "2025-01-01"
    assert str(out_json.loc[0, "ds"].date()) == "2025-01-01"

    with pytest.raises(ValueError):
        pipeline._load_dataset_from_source(
            object(),
            input_method="db_sql",
            dataframe_backend="pandas",
            dataset_schema="dataset",
            dataset_table="tbl",
            dataset_where=None,
            dataset_sql="",
            dataset_path=None,
        )

    with pytest.raises(FileNotFoundError):
        pipeline._load_dataset_from_source(
            object(),
            input_method="csv",
            dataframe_backend="pandas",
            dataset_schema="dataset",
            dataset_table="tbl",
            dataset_where=None,
            dataset_sql=None,
            dataset_path=str(tmp_path / "missing.csv"),
        )


def test_load_dataset_from_source_backend_specific_branches(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("unique_id,ds,y\nA,2025-01-01,1.0\n", encoding="utf-8")

    class _FakePolarsFrame:
        def to_pandas(self):
            return pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})

    class _FakeDaskFrame:
        def compute(self):
            return pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})

    class _FakeSparkDF:
        def toPandas(self):
            return pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})

    class _FakeSparkRead:
        def option(self, *_args, **_kwargs):
            return self

        def csv(self, *_args, **_kwargs):
            return _FakeSparkDF()

        def parquet(self, *_args, **_kwargs):
            return _FakeSparkDF()

        def json(self, *_args, **_kwargs):
            return _FakeSparkDF()

    class _FakeSparkSession:
        def __init__(self):
            self.read = _FakeSparkRead()

    class _FakeSparkBuilder:
        def appName(self, _name):
            return self

        def getOrCreate(self):
            return _FakeSparkSession()

    fake_polars = types.SimpleNamespace(read_csv=lambda _path: _FakePolarsFrame())
    fake_dask = types.SimpleNamespace(read_csv=lambda _path: _FakeDaskFrame())
    fake_dask_pkg = types.SimpleNamespace(dataframe=fake_dask)
    fake_spark = types.SimpleNamespace(
        SparkSession=types.SimpleNamespace(getActiveSession=lambda: None, builder=_FakeSparkBuilder())
    )
    fake_ray_dataset = types.SimpleNamespace(
        read_csv=lambda _path: types.SimpleNamespace(
            to_pandas=lambda: pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "y": [1.0]})
        )
    )
    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    fake_ray.data = fake_ray_dataset

    monkeypatch.setitem(sys.modules, "polars", fake_polars)
    monkeypatch.setitem(sys.modules, "dask", fake_dask_pkg)
    monkeypatch.setitem(sys.modules, "dask.dataframe", fake_dask)
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake_spark)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.data", fake_ray_dataset)

    for backend in ["polars", "dask", "spark", "ray"]:
        out, _source = pipeline._load_dataset_from_source(
            object(),
            input_method="csv",
            dataframe_backend=backend,
            dataset_schema="dataset",
            dataset_table="tbl",
            dataset_where=None,
            dataset_sql=None,
            dataset_path=str(csv_path),
        )
        assert out["unique_id"].tolist() == ["A"]


def test_build_step_metrics_and_other_eval_helpers() -> None:
    merged = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A", "B", "B"],
            "ds": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01", "2025-01-02"]),
            "y": [1.0, 2.0, 3.0, 2.0, None],
            "model_x": [1.0, 2.5, 2.0, 1.0, 5.0],
        }
    )

    rows = pipeline._build_step_split_metrics(merged, "model_x", step_eval_size=2)
    assert [r["step_label"] for r in rows] == ["1-2", "3"]
    assert rows[0]["n"] == 3

    frame = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": ["2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00", "bad"],
            "y": [1, 2, 3],
        }
    )
    norm = pipeline._normalize_eval_key_frame(frame)
    assert len(norm) == 1

    model_col = pipeline._resolve_forecast_model_col(
        pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "cutoff": ["2024-12-31"], "Model": [1.0]})
    )
    assert model_col == "Model"
    with pytest.raises(ValueError):
        pipeline._resolve_forecast_model_col(pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"]}))

    safe_metrics = pipeline._safe_metric_values({"mae": "1.5", "rmse": "bad"})
    assert safe_metrics == {"mae": 1.5, "rmse": None}
    assert pipeline._safe_timestamp_str(pd.Series([], dtype=object)) is None
    assert pipeline._safe_timestamp_str(pd.Series(["bad", "2025-01-02"])) == "2025-01-02 00:00:00"


def test_sanitize_model_input_empty_after_filtering_raises() -> None:
    df = pd.DataFrame({"unique_id": [None], "ds": ["bad"], "y": [None]})
    with pytest.raises(ValueError, match="dataset is empty"):
        pipeline._sanitize_model_input(df)
