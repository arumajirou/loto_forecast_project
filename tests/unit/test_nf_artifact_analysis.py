from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from loto_forecast.analysis import nf_artifact_analysis as mod


def _make_sqlite_table(db_path: Path, table: str, frame: pd.DataFrame) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        frame.to_sql(table, conn, index=False, if_exists="replace")
    finally:
        conn.close()


def test_basic_artifact_file_helpers_and_guess_dataset_id(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run_auto_dataset_loto_y_ts_unified_20260226_091800_80e03c113c"
    artifact_dir.mkdir()
    json_path = artifact_dir / "meta.json"
    pickle_path = artifact_dir / "configuration.pkl"
    ckpt_a = artifact_dir / "a.ckpt"
    ckpt_b = artifact_dir / "z.ckpt"

    json_path.write_text(json.dumps({"h": 3}), encoding="utf-8")
    pickle_path.write_bytes(pickle.dumps({"foo": "bar"}))
    ckpt_a.write_text("x", encoding="utf-8")
    ckpt_b.write_text("y", encoding="utf-8")

    assert mod._read_json(json_path) == {"h": 3}
    assert mod._read_json(artifact_dir / "missing.json") == {}
    assert mod._read_pickle(pickle_path) == {"foo": "bar"}
    assert mod._read_pickle(artifact_dir / "missing.pkl") is None
    assert mod._find_ckpt(artifact_dir) == ckpt_b
    assert mod.guess_dataset_id_from_artifact_dir(artifact_dir) == "dataset_loto_y_ts_unified"

    with pytest.raises(FileNotFoundError):
        mod._find_ckpt(tmp_path / "empty")


def test_safe_to_datetime_standardize_and_metrics() -> None:
    good_df = pd.DataFrame(
        {
            "series": ["A", "A"],
            "date": ["2025-01-02", "2025-01-01"],
            "target": ["2.0", "1.0"],
        }
    )
    df = pd.DataFrame(
        {
            "series": ["A", "A", "A"],
            "date": ["2025-01-02", "2025-01-01", "bad"],
            "target": ["2.0", "1.0", "3.0"],
        }
    )

    out = mod.standardize_to_neuralforecast_format(good_df)

    assert list(out.columns) == ["ds", "y", "unique_id"]
    assert out["unique_id"].tolist() == ["A", "A"]
    assert out["y"].tolist() == [1.0, 2.0]
    assert mod.safe_to_datetime("2025-01-01") is not None
    assert mod.safe_to_datetime(None) is None
    assert mod.safe_to_datetime("not-a-date") is None
    with pytest.raises(ValueError):
        mod.standardize_to_neuralforecast_format(df)

    metrics = mod.compute_metrics(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert metrics["MAE"] == pytest.approx(0.5)
    assert mod.naive_baseline_last_value(np.array([1.0, 4.0]), h=3).tolist() == [4.0, 4.0, 4.0]


def test_load_artifact_bundle_reads_expected_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    (artifact_dir / "meta.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    (artifact_dir / "configuration.pkl").write_bytes(pickle.dumps({"class_path": "pkg.mod:Cls"}))
    (artifact_dir / "alias_to_model.pkl").write_bytes(pickle.dumps({"best": {"epoch": 1}}))
    (artifact_dir / "only.ckpt").write_text("checkpoint", encoding="utf-8")

    bundle = mod.load_artifact_bundle(artifact_dir)

    assert bundle.artifact_dir == artifact_dir.resolve()
    assert bundle.ckpt_path.name == "only.ckpt"
    assert bundle.meta["run_id"] == "r1"
    assert bundle.config["class_path"] == "pkg.mod:Cls"
    assert bundle.alias_to_model["best"]["epoch"] == 1


def test_load_dataset_for_analysis_from_explicit_file_and_sqlite(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    bundle = mod.ArtifactBundle(
        artifact_dir=artifact_dir,
        ckpt_path=artifact_dir / "m.ckpt",
        meta={},
        config={},
        alias_to_model=None,
    )
    csv_path = tmp_path / "dataset.csv"
    db_path = tmp_path / "registry.sqlite"
    frame = pd.DataFrame({"ds": ["2025-01-01"], "y": [1.0], "unique_id": ["A"]})
    frame.to_csv(csv_path, index=False)
    _make_sqlite_table(db_path, "dataset_table", frame)

    from_csv = mod.load_dataset_for_analysis(bundle, dataset_path=csv_path)
    from_sqlite = mod.load_dataset_for_analysis(bundle, sqlite_path=db_path, sqlite_table="dataset_table")

    assert from_csv.to_dict(orient="records") == frame.to_dict(orient="records")
    assert from_sqlite.to_dict(orient="records") == frame.to_dict(orient="records")


def test_load_dataset_for_analysis_guesses_sqlite_table_from_artifact_name(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run_x_dataset_target_series_20260226_091800_abcdef"
    artifact_dir.mkdir()
    bundle = mod.ArtifactBundle(
        artifact_dir=artifact_dir,
        ckpt_path=artifact_dir / "m.ckpt",
        meta={},
        config={},
        alias_to_model=None,
    )
    db_path = tmp_path / "registry.sqlite"
    frame = pd.DataFrame({"ds": ["2025-01-01"], "y": [1.0], "unique_id": ["A"]})
    _make_sqlite_table(db_path, "dataset_target_series", frame)

    out = mod.load_dataset_for_analysis(bundle, sqlite_path=db_path)

    assert out.to_dict(orient="records") == frame.to_dict(orient="records")


def test_load_dataset_for_analysis_raises_when_no_source_found(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    bundle = mod.ArtifactBundle(
        artifact_dir=artifact_dir,
        ckpt_path=artifact_dir / "m.ckpt",
        meta={},
        config={},
        alias_to_model=None,
    )

    with pytest.raises(FileNotFoundError):
        mod.load_dataset_for_analysis(bundle, sqlite_path=tmp_path / "missing.sqlite")


def test_try_import_class_supports_colon_and_dot() -> None:
    assert mod._try_import_class("pathlib:Path") is Path
    assert mod._try_import_class("pathlib.Path") is Path


def test_load_model_from_bundle_prefers_alias_to_model() -> None:
    obj = SimpleNamespace()
    bundle = mod.ArtifactBundle(
        artifact_dir=Path("/tmp/artifact"),
        ckpt_path=Path("/tmp/artifact/model.ckpt"),
        meta={},
        config={},
        alias_to_model={"best": obj},
    )

    loaded = mod.load_model_from_bundle(bundle)

    assert loaded.obj is obj
    assert loaded.extra["alias"] == "best"


def test_load_model_from_bundle_uses_load_from_checkpoint(monkeypatch, tmp_path: Path) -> None:
    class FakeCls:
        @classmethod
        def load_from_checkpoint(cls, path: str, **kwargs):
            return {"path": path, "kwargs": kwargs}

    monkeypatch.setattr(mod, "_try_import_class", lambda _path: FakeCls)
    monkeypatch.setattr(mod, "infer_model_properties", lambda obj: ("lightning", "Fake", {"loaded": obj}))

    bundle = mod.ArtifactBundle(
        artifact_dir=tmp_path,
        ckpt_path=tmp_path / "model.ckpt",
        meta={},
        config={"class_path": "fake.module:FakeCls", "init_kwargs": {"alpha": 1}},
        alias_to_model=None,
    )

    loaded = mod.load_model_from_bundle(bundle)

    assert loaded.kind == "lightning"
    assert loaded.extra["class_path"] == "fake.module:FakeCls"
    assert loaded.extra["loaded"]["kwargs"] == {"alpha": 1}


def test_load_model_from_bundle_loads_state_dict_when_instantiating(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeCls:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def load_state_dict(self, state_dict, strict=False):
            captured["state_dict"] = state_dict
            captured["strict"] = strict

    monkeypatch.setattr(mod, "_try_import_class", lambda _path: FakeCls)
    monkeypatch.setattr(mod, "_load_ckpt_raw", lambda _path: {"state_dict": {"x": 1}})
    monkeypatch.setattr(mod, "infer_model_properties", lambda obj: ("torch", "Fake", {"obj_type": type(obj).__name__}))

    bundle = mod.ArtifactBundle(
        artifact_dir=tmp_path,
        ckpt_path=tmp_path / "model.ckpt",
        meta={},
        config={"class_path": "fake.module.FakeCls", "init_kwargs": {"beta": 2}},
        alias_to_model=None,
    )

    loaded = mod.load_model_from_bundle(bundle)

    assert loaded.kind == "torch"
    assert loaded.extra["class_path"] == "fake.module.FakeCls"
    assert captured["kwargs"] == {"beta": 2}
    assert captured["state_dict"] == {"x": 1}
    assert captured["strict"] is False


def test_load_model_from_bundle_falls_back_to_raw_checkpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mod, "_load_ckpt_raw", lambda _path: {"epoch": 3, "state_dict": {"w": 1}})
    monkeypatch.setattr(mod, "_try_import_class", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))

    bundle = mod.ArtifactBundle(
        artifact_dir=tmp_path,
        ckpt_path=tmp_path / "model.ckpt",
        meta={"model_name": "FallbackModel"},
        config={},
        alias_to_model=None,
    )

    loaded = mod.load_model_from_bundle(bundle)

    assert loaded.kind == "unknown"
    assert loaded.model_name == "FallbackModel"
    assert loaded.obj["epoch"] == 3


def test_infer_model_properties_for_checkpoint_dict() -> None:
    kind, name, extra = mod.infer_model_properties({"epoch": 2, "state_dict": {"a": 1}})

    assert kind == "ckpt_dict"
    assert name == "dict"
    assert extra["epoch"] == 2
    assert extra["state_dict_keys_sample"] == ["a"]


def test_predict_helpers_and_holdout_evaluation_cover_prediction_paths() -> None:
    class FakePredictModel:
        def predict(self, df=None):
            assert isinstance(df, pd.DataFrame)
            return np.array([4.0, 5.0])

    df_nf = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": pd.date_range("2025-01-01", periods=8, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 4.0, 5.0],
        }
    )
    loaded = mod.LoadedModel(obj=FakePredictModel(), kind="unknown", model_name="ModelX", extra={})

    result = mod.evaluate_single_series_holdout(loaded, df_nf, h=2, test_size=2, freq="D")

    assert result["uid"] == "A"
    assert result["used_eval_len"] == 2
    assert result["metrics"]["MAE"] == pytest.approx(0.0)
    assert result["baseline_metrics"]["MAE"] >= 0.0
    assert list(result["pred_df"]["ds"]) == list(pd.date_range("2025-01-07", periods=2, freq="D"))
    assert result["ljung_box"]["available"] is True
    assert result["diebold_mariano_vs_naive"]["available"] is False


def test_extract_point_forecast_and_build_future_ds_edge_cases() -> None:
    forecast_df = pd.DataFrame({"unique_id": ["A"], "ds": ["2025-01-01"], "ModelY": [1.5]})
    out_df = mod.extract_point_forecast(forecast_df, model_name_hint="ModelY")
    out_arr = mod.extract_point_forecast(np.array([1.0, 2.0]))
    out_dict = mod.extract_point_forecast({"prediction": [3.0]})

    assert out_df["y_hat"].tolist() == [1.5]
    assert out_arr["unique_id"].tolist() == ["series_0", "series_0"]
    assert out_dict["y_hat"].tolist() == [3.0]
    assert mod.build_future_ds(pd.Timestamp("2025-01-01"), 2) == list(pd.date_range("2025-01-02", periods=2))
    assert all(pd.isna(x) for x in mod.build_future_ds(pd.NaT, 2))


def test_diebold_mariano_large_sample_returns_statistics() -> None:
    e1 = np.linspace(0.1, 1.2, 12)
    e2 = np.linspace(0.2, 1.4, 12)

    out = mod.diebold_mariano_test(e1, e2, h=2, power=2)

    assert out["available"] is True
    assert out["n"] == 12
    assert "dm_t" in out
