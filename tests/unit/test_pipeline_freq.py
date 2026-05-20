from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from loto_forecast.orchestration import pipeline


def test_train_passes_freq_from_model_params(monkeypatch, tmp_path: Path):
    captured: dict = {"freq": None, "upsert_meta": []}
    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
        }
    )

    def _fake_upsert(*args, **kwargs):
        if len(args) >= 4 and isinstance(args[3], dict):
            captured["upsert_meta"].append(dict(args[3]))

    def _fake_train_automodel(df, model_name, h, freq, run_id, model_params):
        captured["freq"] = str(freq)
        return SimpleNamespace(
            run_id=run_id,
            model_name=model_name,
            artifact_path=tmp_path / run_id,
            exog={},
        )

    monkeypatch.setattr(pipeline, "setup_logging", lambda _rid: tmp_path / "run.log")
    monkeypatch.setattr(pipeline, "make_engine", lambda: object())
    monkeypatch.setattr(pipeline, "read_timeseries", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(pipeline, "prepare_dataset", lambda raw: raw.copy())
    monkeypatch.setattr(pipeline, "_sanitize_model_input", lambda raw: raw.copy())
    monkeypatch.setattr(pipeline, "upsert_model_run", _fake_upsert)
    monkeypatch.setattr(pipeline, "mark_model_run_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("loto_forecast.models.neuralforecast_model.train_automodel", _fake_train_automodel)

    out = pipeline.train(
        model_name="AutoNHITS",
        h=2,
        model_params={
            "freq": "W",
            "dataset_schema": "dataset",
            "dataset_table": "loto_y_ts",
            "target_unique_id": "N1",
        },
        run_id="run_freq_override",
    )

    assert out["run_id"] == "run_freq_override"
    assert captured["freq"] == "W"
    assert any(str(m.get("freq")) == "W" for m in captured["upsert_meta"])


def test_train_loto_ts_type_forces_target_unique_id_none(monkeypatch, tmp_path: Path):
    captured: dict = {"upsert_meta": [], "fit_df": None}
    df = pd.DataFrame(
        {
            "loto": ["bingo5", "bingo5", "bingo5", "bingo5"],
            "unique_id": ["N1", "N2", "N1", "N2"],
            "ts_type": ["raw", "raw", "raw", "raw"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "y": [1.0, 3.0, 3.0, 5.0],
        }
    )

    def _fake_upsert(*args, **kwargs):
        if len(args) >= 4 and isinstance(args[3], dict):
            captured["upsert_meta"].append(dict(args[3]))

    def _fake_train_automodel(df, model_name, h, freq, run_id, model_params):
        captured["fit_df"] = df.copy()
        return SimpleNamespace(
            run_id=run_id,
            model_name=model_name,
            artifact_path=tmp_path / run_id,
            exog={},
        )

    monkeypatch.setattr(pipeline, "setup_logging", lambda _rid: tmp_path / "run.log")
    monkeypatch.setattr(pipeline, "make_engine", lambda: object())
    monkeypatch.setattr(pipeline, "read_timeseries", lambda *_args, **_kwargs: df.copy())
    monkeypatch.setattr(pipeline, "prepare_dataset", lambda raw: raw.copy())
    monkeypatch.setattr(pipeline, "_sanitize_model_input", lambda raw: raw.copy())
    monkeypatch.setattr(pipeline, "upsert_model_run", _fake_upsert)
    monkeypatch.setattr(pipeline, "mark_model_run_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("loto_forecast.models.neuralforecast_model.train_automodel", _fake_train_automodel)

    out = pipeline.train(
        model_name="AutoNHITS",
        h=2,
        model_params={
            "freq": "D",
            "dataset_schema": "dataset",
            "dataset_table": "loto_y_ts",
            "group_by_mode": "loto_ts_type",
            "target_loto": "bingo5",
            "target_unique_id": "N1",
            "target_ts_type": "raw",
        },
        run_id="run_group_mode_loto_ts_type",
    )

    assert out["run_id"] == "run_group_mode_loto_ts_type"
    assert any((m.get("data_selection", {}) or {}).get("target_unique_id") == [] for m in captured["upsert_meta"])
    fit_df = captured["fit_df"]
    assert isinstance(fit_df, pd.DataFrame)
    assert set(fit_df["unique_id"].astype(str).unique().tolist()) == {"bingo5__raw"}


def test_train_raises_when_group_mode_requires_unique_id(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pipeline, "setup_logging", lambda _rid: tmp_path / "run.log")
    with pytest.raises(ValueError, match="group_by_mode=loto_unique_id_ts_type requires non-empty target_unique_id"):
        pipeline.train(
            model_name="AutoNHITS",
            h=2,
            model_params={
                "group_by_mode": "loto_unique_id_ts_type",
                "target_unique_id": "",
            },
            run_id="run_group_mode_requires_uid",
        )
