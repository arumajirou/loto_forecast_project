import argparse
from pathlib import Path

import pandas as pd

from loto_forecast import cli
from loto_forecast.models import neuralforecast_model as nfm


class _FakeNF:
    def __init__(self):
        self.models = [object(), object()]

    def save(self, path: str, overwrite: bool, save_dataset: bool):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "nhits_0.ckpt").write_bytes(b"ckpt")
        (p / "nhits_0.pkl").write_bytes(b"pkl")
        if save_dataset:
            (p / "dataset.pkl").write_bytes(b"dataset")

    def predict_insample(self, step_size: int = 1):
        return pd.DataFrame({"unique_id": [1], "ds": ["2020-01-01"], "yhat": [1.0], "step_size": [step_size]})


def test_save_load_analyze_model_bundle(monkeypatch, tmp_path: Path):
    src = tmp_path / "src_run"
    src.mkdir(parents=True, exist_ok=True)
    (src / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(nfm, "load_model", lambda path, load_kwargs=None: _FakeNF())

    out = nfm.save_load_analyze_model_bundle(
        run_id="run_x",
        source_dir=src,
        save_path=str(tmp_path / "saved"),
        run_save=True,
        run_load=True,
        run_analyze=True,
        save_dataset=True,
        save_overwrite=True,
        load_check_predict=True,
        insample_step_size=2,
    )

    assert out["run_id"] == "run_x"
    assert out["save"]["ok"] is True
    assert out["load"]["ok"] is True
    assert out["load"]["model_count"] == 2
    assert out["load"]["predict_insample"]["ok"] is True
    assert out["analyze"]["file_count"] >= 2


def test_cmd_model_save_load_analyze_resolves_base_dirs(monkeypatch, tmp_path: Path):
    run_id = "run_x"
    source_root = tmp_path / "artifacts"
    source_dir = source_root / run_id
    source_dir.mkdir(parents=True, exist_ok=True)
    save_root = tmp_path / "saved_models"
    (save_root / run_id).mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def _fake_bundle(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(nfm, "save_load_analyze_model_bundle", _fake_bundle)
    args = argparse.Namespace(
        run_id=run_id,
        source_path=str(source_root),
        save_path=str(save_root),
        run_save=False,
        run_load=True,
        run_analyze=False,
        save_dataset=False,
        save_overwrite=True,
        load_check_predict=False,
        insample_step_size=1,
        save_kwargs_json=None,
        load_kwargs_json=None,
        predict_insample_kwargs_json=None,
    )

    cli.cmd_model_save_load_analyze(args)

    assert Path(captured["source_dir"]) == source_dir
    assert Path(str(captured["save_path"])) == (save_root / run_id)

    captured.clear()
    args.save_path = str(save_root / "{run_id}")
    cli.cmd_model_save_load_analyze(args)
    assert Path(str(captured["save_path"])) == (save_root / run_id)
