import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from loto_forecast.models import neuralforecast_model as nfm


def test_load_neuralforecast_runtime_applies_safe_topk_patch(monkeypatch):
    nf = types.ModuleType("neuralforecast")
    losses = types.ModuleType("neuralforecast.losses")
    losses_pt = types.ModuleType("neuralforecast.losses.pytorch")

    class DummyNeuralForecast:
        pass

    class DummyMAE:
        pass

    nf.NeuralForecast = DummyNeuralForecast
    losses_pt.MAE = DummyMAE
    monkeypatch.setitem(sys.modules, "neuralforecast", nf)
    monkeypatch.setitem(sys.modules, "neuralforecast.losses", losses)
    monkeypatch.setitem(sys.modules, "neuralforecast.losses.pytorch", losses_pt)

    calls = {"n": 0}

    def _fake_apply():
        calls["n"] += 1

    monkeypatch.setattr("loto_forecast.patches.neuralforecast_autoformer_safe_topk.apply", _fake_apply)

    NeuralForecast, MAE = nfm._load_neuralforecast_runtime()

    assert calls["n"] == 1
    assert NeuralForecast is DummyNeuralForecast
    assert MAE is DummyMAE


def test_build_automodel_does_not_pass_random_seed(monkeypatch):
    captured: dict = {}

    class DummyAutoModel:
        def __init__(self, h, loss, num_samples, backend):
            captured.update({"h": h, "loss": loss, "num_samples": num_samples, "backend": backend})

    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (object(), object()))
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyAutoModel)

    _ = nfm.build_automodel(
        model_name="AutoNHITS",
        h=14,
        exog={},
        backend="optuna",
        num_samples=3,
        seed=42,
    )

    assert captured["h"] == 14
    assert captured["backend"] == "optuna"
    assert captured["num_samples"] == 3
    assert "random_seed" not in captured


def test_build_automodel_skips_unsupported_exog_kwargs(monkeypatch):
    captured: dict = {}

    class DummyAutoModel:
        def __init__(self, h, loss, num_samples, backend):
            captured.update({"h": h, "loss": loss, "num_samples": num_samples, "backend": backend})

    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (object(), object()))
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyAutoModel)

    _ = nfm.build_automodel(
        model_name="AutoNHITS",
        h=14,
        exog={"hist_exog": ["a"], "futr_exog": ["b"], "stat_exog": ["c"]},
        backend="optuna",
        num_samples=3,
        seed=42,
    )

    assert "hist_exog_list" not in captured
    assert "futr_exog_list" not in captured
    assert "stat_exog_list" not in captured


def test_build_automodel_forces_valid_loss_to_loss(monkeypatch):
    captured: dict = {}

    class DummyAutoModel:
        def __init__(self, h, loss, valid_loss, num_samples, backend):
            captured.update(
                {
                    "h": h,
                    "loss": loss,
                    "valid_loss": valid_loss,
                    "num_samples": num_samples,
                    "backend": backend,
                }
            )

    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (object(), object()))
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyAutoModel)
    monkeypatch.setattr(nfm, "_build_loss", lambda name: f"loss::{name}")

    _ = nfm.build_automodel(
        model_name="AutoNHITS",
        h=7,
        exog={},
        backend="optuna",
        num_samples=1,
        loss_name="MAE",
        valid_loss_name="RMSE",
    )

    assert captured["loss"] == "loss::MAE"
    assert captured["valid_loss"] == "loss::MAE"


def test_train_automodel_accepts_random_seed_alias(monkeypatch, tmp_path: Path):
    captured_seed = {"value": None}

    class FakeNF:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame):
            assert len(df) > 0

        def save(self, path: str, overwrite: bool, save_dataset: bool):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    def _fake_build_automodel(**kwargs):
        captured_seed["value"] = kwargs.get("seed")
        return object()

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(nfm, "infer_exog_columns", lambda _df: {})
    monkeypatch.setattr(nfm, "build_automodel", _fake_build_automodel)
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
        }
    )
    out = nfm.train_automodel(
        df=df,
        model_name="AutoNHITS",
        h=2,
        run_id="run_seed_alias",
        model_params={"backend": "optuna", "num_samples": 1, "random_seed": 9},
    )

    assert out.run_id == "run_seed_alias"
    assert captured_seed["value"] == 9


def test_train_automodel_forces_valid_loss_and_syncs_local_scalers(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeNF:
        def __init__(self, models, freq, **kwargs):
            captured["nf_kwargs"] = dict(kwargs)
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame, **kwargs):
            assert len(df) > 0

        def save(self, path: str, overwrite: bool, save_dataset: bool):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    def _fake_build_automodel(**kwargs):
        captured["build_kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(nfm, "infer_exog_columns", lambda _df: {})
    monkeypatch.setattr(nfm, "build_automodel", _fake_build_automodel)
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
        }
    )
    _ = nfm.train_automodel(
        df=df,
        model_name="AutoNHITS",
        h=2,
        run_id="run_scaler_sync",
        model_params={
            "backend": "optuna",
            "num_samples": 1,
            "loss_name": "MAE",
            "valid_loss_name": "RMSE",
            "local_scaler_type": "robust",
            "local_static_scaler_type": "standard",
        },
    )

    assert captured["build_kwargs"]["valid_loss_name"] == "MAE"
    assert captured["nf_kwargs"]["local_scaler_type"] == "robust"
    assert captured["nf_kwargs"]["local_static_scaler_type"] == "robust"


def test_prepare_nf_frames_fills_missing_and_reduces_columns():
    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "y": [1.0, 2.0, 3.0],
            "hist_a": [None, 1.0, None],
            "feat_b": [10.0, None, 12.0],
            "text_col": ["a", "b", "c"],
        }
    )
    fit_df, futr_df, exog = nfm.prepare_nf_frames(
        df, exog={"hist_exog": ["hist_a"], "futr_exog": ["feat_b"], "stat_exog": ["text_col"]}
    )
    assert set(fit_df.columns) == {"unique_id", "ds", "y", "hist_a", "feat_b"}
    assert int(fit_df[["hist_a", "feat_b"]].isna().sum().sum()) == 0
    assert "text_col" not in exog["stat_exog"]
    assert futr_df is None


def test_prepare_nf_frames_adds_missing_futr_cols():
    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1"],
            "ds": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "y": [1.0, 2.0],
            "feat_needed": [3.0, 4.0],
        }
    )
    futr = pd.DataFrame(
        {
            "unique_id": ["N1"],
            "ds": pd.to_datetime(["2025-01-03"]),
        }
    )
    _, futr_out, exog = nfm.prepare_nf_frames(
        df, exog={"futr_exog": ["feat_needed"], "hist_exog": [], "stat_exog": []}, futr_df=futr
    )
    assert exog["futr_exog"] == ["feat_needed"]
    assert "feat_needed" in futr_out.columns
    assert float(futr_out.loc[0, "feat_needed"]) == 3.5


def test_validate_runtime_kwargs_rejects_unknown_and_blocked():
    out = nfm.validate_runtime_kwargs(
        {
            "nf_fit_kwargs": {"df": "bad", "unknown_key": 1},
            "nf_load_kwargs": {"path": "/tmp/x"},
        }
    )
    assert out["ok"] is False
    assert any("nf_fit_kwargs.df is not allowed" in e for e in out["errors"])
    assert any("nf_fit_kwargs: unknown option 'unknown_key'" in e for e in out["errors"])
    assert any("nf_load_kwargs.path is not allowed" in e for e in out["errors"])


def test_train_automodel_applies_nf_runtime_kwargs(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeNF:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame, **kwargs):
            captured["fit_kwargs"] = dict(kwargs)
            assert len(df) > 0

        def cross_validation(self, df: pd.DataFrame, **kwargs):
            captured["cv_kwargs"] = dict(kwargs)
            return pd.DataFrame({"unique_id": ["N1"], "ds": pd.to_datetime(["2024-01-03"]), "AutoNHITS": [1.0]})

        def save(self, path: str, **kwargs):
            captured["save_kwargs"] = dict(kwargs)
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(nfm, "infer_exog_columns", lambda _df: {})
    monkeypatch.setattr(nfm, "build_automodel", lambda **_kwargs: object())
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
        }
    )
    out = nfm.train_automodel(
        df=df,
        model_name="AutoNHITS",
        h=2,
        run_id="run_nf_kwargs",
        model_params={
            "backend": "optuna",
            "num_samples": 1,
            "run_cross_validation": True,
            "nf_fit_kwargs": {"val_size": 1, "verbose": True},
            "nf_cross_validation_kwargs": {"n_windows": 1, "step_size": 1},
            "nf_save_kwargs": {"save_dataset": True, "overwrite": False},
        },
    )

    assert out.run_id == "run_nf_kwargs"
    assert captured["fit_kwargs"]["val_size"] == 1
    assert captured["fit_kwargs"]["verbose"] is True
    assert captured["cv_kwargs"]["n_windows"] == 1
    assert captured["save_kwargs"]["save_dataset"] is True
    assert captured["save_kwargs"]["overwrite"] is False

    meta = json.loads((tmp_path / "run_nf_kwargs" / "meta.json").read_text(encoding="utf-8"))
    assert meta["nf_runtime_kwargs"]["nf_fit_kwargs"]["val_size"] == 1
    assert meta["cross_validation"]["enabled"] is True


def test_train_automodel_rejects_explicit_unsupported_exog(monkeypatch, tmp_path: Path):
    class FakeNF:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame, **kwargs):
            assert len(df) > 0

        def save(self, path: str, **kwargs):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(nfm, "build_automodel", lambda **_kwargs: object())
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
            "feat_x": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(ValueError) as exc:
        _ = nfm.train_automodel(
            df=df,
            model_name="AutoPatchTST",
            h=2,
            run_id="run_bad_exog",
            model_params={
                "backend": "optuna",
                "num_samples": 1,
                "strict_exog": True,
                "futr_exog_list": ["feat_x"],
            },
        )
    assert "does not support futr_exog" in str(exc.value)


def test_train_automodel_resolves_static_df_from_column_list(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeNF:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame, **kwargs):
            captured["fit_kwargs"] = dict(kwargs)
            assert len(df) > 0

        def save(self, path: str, **kwargs):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(
        nfm, "infer_exog_columns", lambda _df: {"stat_exog": ["stat_profile"], "futr_exog": [], "hist_exog": []}
    )
    monkeypatch.setattr(nfm, "build_automodel", lambda **_kwargs: object())
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N2"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "y": [1.0, 2.0, 3.0],
            "stat_profile": [10.0, 10.0, 20.0],
        }
    )
    _ = nfm.train_automodel(
        df=df,
        model_name="AutoNHITS",
        h=2,
        run_id="run_static_df_auto",
        model_params={
            "backend": "optuna",
            "num_samples": 1,
            "nf_fit_kwargs": {"static_df": ["stat_profile"]},
        },
    )

    assert "static_df" in captured["fit_kwargs"]
    static_df = captured["fit_kwargs"]["static_df"]
    assert list(static_df.columns) == ["unique_id", "stat_profile"]
    assert int(static_df.shape[0]) == 2


def test_train_automodel_autofills_required_n_series(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class DummyMultivariateModel:
        def __init__(self, h, loss, valid_loss, n_series, backend="ray", num_samples=1):  # noqa: ANN001
            self.h = h
            self.loss = loss
            self.valid_loss = valid_loss
            self.n_series = n_series
            self.backend = backend
            self.num_samples = num_samples

    class FakeAdapter:
        def validate(self, model_name: str, model_params: dict[str, object]) -> dict[str, object]:
            captured["validated_model"] = model_name
            captured["validated_params"] = dict(model_params)
            if "n_series" not in model_params:
                return {
                    "ok": False,
                    "errors": ["missing required model param for AutoMLPMultivariate: n_series"],
                    "warnings": [],
                }
            return {"ok": True, "errors": [], "warnings": []}

    class FakeNF:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame, **kwargs):
            assert len(df) > 0

        def save(self, path: str, **kwargs):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    def _fake_build_automodel(**kwargs):
        captured["build_kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyMultivariateModel)
    monkeypatch.setattr("loto_forecast.models.registry.get_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(nfm, "infer_exog_columns", lambda _df: {})
    monkeypatch.setattr(nfm, "build_automodel", _fake_build_automodel)
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N2", "N2"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = nfm.train_automodel(
        df=df,
        model_name="AutoMLPMultivariate",
        h=2,
        run_id="run_auto_n_series",
        model_params={"backend": "ray", "num_samples": 1},
    )

    assert out.run_id == "run_auto_n_series"
    assert captured["validated_model"] == "AutoMLPMultivariate"
    assert captured["validated_params"]["n_series"] == 2
    assert captured["build_kwargs"]["model_kwargs"]["n_series"] == 2


def test_build_automodel_applies_h1_config_overrides_for_auto_models(monkeypatch):
    captured: dict = {}

    class DummyAutoModel:
        @classmethod
        def get_default_config(cls, h, backend, n_series=None):  # noqa: ANN001
            if backend == "optuna":
                return lambda _trial: {"input_size": 1, "step_size": 1}
            return {"input_size": 1, "step_size": 1}

        def __init__(self, h, loss, valid_loss, num_samples, backend, config):
            captured["h"] = h
            captured["backend"] = backend
            captured["config"] = config

    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (object(), object()))
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyAutoModel)

    _ = nfm.build_automodel(
        model_name="AutoInformer",
        h=1,
        exog={},
        backend="optuna",
        num_samples=1,
    )

    cfg = captured["config"]
    assert callable(cfg)
    trial_cfg = cfg(object())
    assert trial_cfg["input_size"] == 2
    assert trial_cfg["step_size"] == 1


def test_build_automodel_applies_h1_timesnet_and_timexer_overrides(monkeypatch):
    captured: dict = {}

    class DummyAutoModel:
        @classmethod
        def get_default_config(cls, h, backend, n_series=None):  # noqa: ANN001
            if backend == "optuna":
                return lambda _trial: {"input_size": 1}
            return {"input_size": 1}

        def __init__(self, h, loss, valid_loss, num_samples, backend, config, n_series):
            captured["config"] = config
            captured["n_series"] = n_series

    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (object(), object()))
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyAutoModel)

    _ = nfm.build_automodel(
        model_name="AutoTimesNet",
        h=1,
        exog={},
        backend="optuna",
        num_samples=1,
        model_kwargs={"n_series": 1},
    )
    cfg_timesnet = captured["config"](object())
    assert cfg_timesnet["top_k"] == 1

    _ = nfm.build_automodel(
        model_name="AutoTimeXer",
        h=1,
        exog={},
        backend="optuna",
        num_samples=1,
        model_kwargs={"n_series": 1},
    )
    cfg_timexer = captured["config"](object())
    assert cfg_timexer["patch_len"] == 1


def test_build_automodel_applies_h1_nbeats_stack_override(monkeypatch):
    captured: dict = {}

    class DummyAutoModel:
        @classmethod
        def get_default_config(cls, h, backend, n_series=None):  # noqa: ANN001
            if backend == "optuna":
                return lambda _trial: {"input_size": 1}
            return {"input_size": 1}

        def __init__(self, h, loss, valid_loss, num_samples, backend, config):
            captured["config"] = config

    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (object(), object()))
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyAutoModel)

    _ = nfm.build_automodel(
        model_name="AutoNBEATS",
        h=1,
        exog={},
        backend="optuna",
        num_samples=1,
    )

    cfg = captured["config"](object())
    assert cfg["stack_types"] == ["identity", "identity", "identity"]


def test_train_automodel_autofills_input_size_two_for_decoder_models(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class DummyDecoderModel:
        def __init__(self, h, input_size, decoder_input_size_multiplier=0.5, loss=None, valid_loss=None):  # noqa: ANN001
            self.h = h
            self.input_size = input_size
            self.decoder_input_size_multiplier = decoder_input_size_multiplier
            self.loss = loss
            self.valid_loss = valid_loss

    class FakeAdapter:
        def validate(self, model_name: str, model_params: dict[str, object]) -> dict[str, object]:
            return {"ok": True, "errors": [], "warnings": []}

    class FakeNF:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq

        def fit(self, df: pd.DataFrame, **kwargs):
            assert len(df) > 0

        def save(self, path: str, **kwargs):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "model.bin").write_bytes(b"x")

    def _fake_build_automodel(**kwargs):
        captured["build_kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(nfm.settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(nfm, "_resolve_model_class", lambda _name: DummyDecoderModel)
    monkeypatch.setattr("loto_forecast.models.registry.get_adapter", lambda _name: FakeAdapter())
    monkeypatch.setattr(nfm, "infer_exog_columns", lambda _df: {})
    monkeypatch.setattr(nfm, "build_automodel", _fake_build_automodel)
    monkeypatch.setattr(nfm, "_load_neuralforecast_runtime", lambda: (FakeNF, object()))

    df = pd.DataFrame(
        {
            "unique_id": ["N1", "N1", "N1"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [1.0, 2.0, 3.0],
        }
    )
    _ = nfm.train_automodel(
        df=df,
        model_name="Autoformer",
        h=1,
        run_id="run_auto_input_size_h1",
        model_params={"backend": "optuna", "num_samples": 1},
    )

    assert captured["build_kwargs"]["model_kwargs"]["input_size"] == 2
