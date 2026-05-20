from loto_forecast.models import registry as reg
from loto_forecast.models.registry import get_adapter


def test_adapter_unknown_model_validation():
    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate("NoSuchModel", {})
    assert out["ok"] is False
    assert any("unsupported" in e for e in out["errors"])


def test_adapter_rejects_unknown_and_type_mismatch(monkeypatch):
    class DummyModel:
        def __init__(self, h=1):
            self.h = h

    monkeypatch.setattr(reg, "_auto_model_names", lambda: ["AutoNHITS"])
    monkeypatch.setattr(reg, "_resolve_model_class", lambda _name: DummyModel)

    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate("AutoNHITS", {"num_samples": "10", "bad_key": 1})
    assert out["ok"] is False
    assert any("unknown model params" in e for e in out["errors"])
    assert any("num_samples must be int > 0" in e for e in out["errors"])


def test_adapter_rejects_unsupported_exog_for_model(monkeypatch):
    class DummyModel:
        def __init__(self, h=1):
            self.h = h

    monkeypatch.setattr(reg, "_auto_model_names", lambda: ["AutoPatchTST"])
    monkeypatch.setattr(reg, "_resolve_model_class", lambda _name: DummyModel)

    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate("AutoPatchTST", {"futr_exog_list": ["feat_x"]})
    assert out["ok"] is False
    assert any("does not support futr_exog_list" in e for e in out["errors"])


def test_adapter_rejects_invalid_runtime_kwargs(monkeypatch):
    class DummyModel:
        def __init__(self, h=1):
            self.h = h

    monkeypatch.setattr(reg, "_auto_model_names", lambda: ["AutoNHITS"])
    monkeypatch.setattr(reg, "_resolve_model_class", lambda _name: DummyModel)

    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate("AutoNHITS", {"nf_fit_kwargs": {"unknown_key": 1}})
    assert out["ok"] is False
    assert any("unknown option 'unknown_key'" in e for e in out["errors"])


def test_adapter_rejects_mismatched_loss_and_scaler_pairs(monkeypatch):
    class DummyModel:
        def __init__(self, h=1):
            self.h = h

    monkeypatch.setattr(reg, "_auto_model_names", lambda: ["AutoNHITS"])
    monkeypatch.setattr(reg, "_resolve_model_class", lambda _name: DummyModel)

    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate(
        "AutoNHITS",
        {
            "loss_name": "MAE",
            "valid_loss_name": "RMSE",
            "local_scaler_type": "robust",
            "local_static_scaler_type": "standard",
        },
    )
    assert out["ok"] is False
    assert any("valid_loss_name must match loss_name" in e for e in out["errors"])
    assert any("local_static_scaler_type must match local_scaler_type" in e for e in out["errors"])


def test_adapter_accepts_freq_reserved_param(monkeypatch):
    class DummyModel:
        def __init__(self, h=1):
            self.h = h

    monkeypatch.setattr(reg, "_auto_model_names", lambda: ["AutoNHITS"])
    monkeypatch.setattr(reg, "_resolve_model_class", lambda _name: DummyModel)

    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate("AutoNHITS", {"freq": "W"})
    assert out["ok"] is True
