from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from loto_forecast.analysis import explain as mod


def test_prepare_validates_required_columns_and_filters() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A", None],
            "ds": ["2025-01-02", "2025-01-02", "bad"],
            "y": [2.0, 3.0, 1.0],
        }
    )
    out = mod._prepare(df)

    assert len(out) == 1
    assert float(out.iloc[0]["y"]) == 3.0

    with pytest.raises(ValueError, match="missing required columns"):
        mod._prepare(pd.DataFrame({"unique_id": ["A"]}))

    with pytest.raises(ValueError, match="dataset is empty"):
        mod._prepare(pd.DataFrame({"unique_id": [None], "ds": ["bad"], "y": [None]}))


def test_exog_granger_screening_and_json_load(monkeypatch, tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["A", "A", "A"],
            "ds": pd.date_range("2025-01-01", periods=3, freq="D"),
            "y": [1.0, 2.0, 3.0],
            "x1": [2.0, 3.0, 4.0],
            "x2": [9.0, 8.0, 7.0],
            "label": ["a", "b", "c"],
        }
    )
    monkeypatch.setattr(mod, "make_engine", lambda: object())
    monkeypatch.setattr(mod, "read_timeseries", lambda *_args, **_kwargs: frame.copy())
    monkeypatch.setattr(mod, "granger_test", lambda _single, y_col, x_col, maxlag: {"1": 0.2, "2": 0.1 if x_col == "x1" else 0.3})

    out = mod.exog_granger_screening(maxlag=2, top_k=1)

    assert out.to_dict(orient="records") == [{"feature": "x1", "min_pvalue": 0.1, "best_lag": 2}]

    json_path = tmp_path / "meta.json"
    json_path.write_text(json.dumps({"h": 3}), encoding="utf-8")
    assert mod.json_load(json_path) == {"h": 3}


def test_neuralforecast_explainability_available_and_unavailable(monkeypatch) -> None:
    fake_module = types.ModuleType("neuralforecast.explainability")
    fake_module.Explainability = object
    monkeypatch.setitem(sys.modules, "neuralforecast.explainability", fake_module)
    assert mod.neuralforecast_explainability("run1").startswith("Explainability is available")

    monkeypatch.delitem(sys.modules, "neuralforecast.explainability", raising=False)
    monkeypatch.setattr(
        __import__("builtins"),
        "__import__",
        lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError("missing"))
        if name == "neuralforecast.explainability"
        else __import__(name, *args, **kwargs),
    )
    assert "import failed" in mod.neuralforecast_explainability("run1")


def test_permutation_importance_exog_with_mocked_model_stack(monkeypatch, tmp_path: Path) -> None:
    run_id = "run_perm"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"h": 2, "exog": {"hist_exog": ["hist_x"], "futr_exog": ["futr_x"]}}), encoding="utf-8")

    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 6,
            "ds": pd.date_range("2025-01-01", periods=6, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "hist_x": [10, 11, 12, 13, 14, 15],
            "futr_x": [20, 21, 22, 23, 24, 25],
        }
    )

    monkeypatch.setattr(mod.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(mod, "make_engine", lambda: object())
    monkeypatch.setattr(mod, "read_timeseries", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(mod, "make_future_df", lambda train_df, h, freq: train_df.tail(h)[["unique_id", "ds"]].copy())
    monkeypatch.setattr(mod, "add_time_features", lambda frame: frame.copy())
    monkeypatch.setattr(mod, "add_cyclical_time_features", lambda frame: frame.copy())
    monkeypatch.setattr(mod, "write_exog_contribution", lambda *_args, **_kwargs: None)

    predict_calls = {"count": 0}

    def _fake_prepare_nf_frames(df, exog, futr_df):
        return df.copy(), futr_df.assign(futr_x=[100, 101]), exog

    def _fake_predict_with_model(_nf, df, futr_df):
        predict_calls["count"] += 1
        target = pd.DataFrame({"unique_id": ["A", "A"], "ds": pd.date_range("2025-01-05", periods=2, freq="D")})
        if predict_calls["count"] == 1:
            target["Model"] = [5.0, 6.0]
        else:
            target["Model"] = [6.0, 7.0]
        return target

    monkeypatch.setattr("loto_forecast.models.neuralforecast_model.load_model", lambda _run_dir: object())
    monkeypatch.setattr("loto_forecast.models.neuralforecast_model.prepare_nf_frames", _fake_prepare_nf_frames)
    monkeypatch.setattr("loto_forecast.models.neuralforecast_model.predict_with_model", _fake_predict_with_model)

    out = mod.permutation_importance_exog(run_id=run_id, n_repeats=2, dataset_df=df)

    assert set(out["feature"]) == {"hist_x", "futr_x"}
    assert (run_dir / "exog_importance.parquet").exists()


def test_permutation_importance_exog_raises_when_train_split_empty(monkeypatch, tmp_path: Path) -> None:
    run_id = "run_short"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"h": 3}), encoding="utf-8")
    df = pd.DataFrame({"unique_id": ["A", "A"], "ds": pd.date_range("2025-01-01", periods=2, freq="D"), "y": [1.0, 2.0]})

    monkeypatch.setattr(mod.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr("loto_forecast.models.neuralforecast_model.load_model", lambda _run_dir: object())

    with pytest.raises(ValueError, match="train split is empty"):
        mod.permutation_importance_exog(run_id=run_id, dataset_df=df)
