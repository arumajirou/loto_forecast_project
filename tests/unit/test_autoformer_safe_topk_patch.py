from __future__ import annotations

import sys
import types

import torch

from loto_forecast.patches import neuralforecast_autoformer_safe_topk as patch_mod


def _install_fake_neuralforecast(monkeypatch):
    nf = types.ModuleType("neuralforecast")
    nf_models = types.ModuleType("neuralforecast.models")
    nf_autoformer = types.ModuleType("neuralforecast.models.autoformer")

    class FakeAutoCorrelation:
        def __init__(self, factor: float = 3.0):
            self.factor = factor

    nf_autoformer.AutoCorrelation = FakeAutoCorrelation

    monkeypatch.setitem(sys.modules, "neuralforecast", nf)
    monkeypatch.setitem(sys.modules, "neuralforecast.models", nf_models)
    monkeypatch.setitem(sys.modules, "neuralforecast.models.autoformer", nf_autoformer)

    return FakeAutoCorrelation


def test_clamp_top_k_bounds():
    assert patch_mod._clamp_top_k(3.0, 0) == 0
    assert patch_mod._clamp_top_k(3.0, 1) == 1
    assert patch_mod._clamp_top_k(3.0, 2) == 2
    assert patch_mod._clamp_top_k(100.0, 2) == 2


def test_apply_monkeypatches_autocorrelation(monkeypatch):
    FakeAutoCorrelation = _install_fake_neuralforecast(monkeypatch)
    monkeypatch.setattr(patch_mod, "_PATCH_APPLIED", False)

    patch_mod.apply()

    autocorr = FakeAutoCorrelation(factor=100.0)
    values = torch.randn(2, 1, 1, 2)
    corr = torch.randn(2, 1, 1, 2)

    out_train = autocorr.time_delay_agg_training(values, corr)
    out_infer = autocorr.time_delay_agg_inference(values, corr)
    out_full = autocorr.time_delay_agg_full(values, corr)

    assert tuple(out_train.shape) == (2, 1, 1, 2)
    assert tuple(out_infer.shape) == (2, 1, 1, 2)
    assert tuple(out_full.shape) == (2, 1, 1, 2)


def test_apply_is_idempotent(monkeypatch):
    FakeAutoCorrelation = _install_fake_neuralforecast(monkeypatch)
    monkeypatch.setattr(patch_mod, "_PATCH_APPLIED", False)

    patch_mod.apply()
    first_fn = FakeAutoCorrelation.time_delay_agg_training

    patch_mod.apply()
    second_fn = FakeAutoCorrelation.time_delay_agg_training

    assert first_fn is second_fn
