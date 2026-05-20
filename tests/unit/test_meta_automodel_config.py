import pytest

from loto_forecast.orchestration.meta_automodel import _normalize_meta_config, _validate_meta_model_arguments


def test_normalize_meta_config_requires_config_name():
    with pytest.raises(ValueError):
        _normalize_meta_config({})


def test_normalize_meta_config_applies_defaults_and_parses_json():
    out = _normalize_meta_config(
        {
            "config_name": "cfg_a",
            "horizon": "35",
            "unified_filter_json": '{"loto":"bingo5","unique_id":"N1","ts_type":"raw"}',
            "unified_group_cols_json": '["loto","unique_id","ts_type"]',
            "unified_group_validate_strict": "true",
            "auto_cls_model": "AutoNHITS",
            "auto_h": "14",
            "model_params_json": '{"num_samples": 20}',
            "auto_config_json": '{"backend":"optuna","num_samples":10}',
            "param_space_json": {"seed": [1, 2]},
            "auto_callbacks_json": '["pkg.cb1", "pkg.cb2"]',
            "auto_num_samples": "11",
            "run_save": "true",
            "run_load": "false",
            "run_analyze": "true",
            "run_predict": "false",
            "recursive_depth": 0,
        }
    )
    assert out["config_name"] == "cfg_a"
    assert out["horizon"] == 35
    assert out["unified_filter_json"]["loto"] == "bingo5"
    assert out["unified_group_cols_json"] == ["loto", "unique_id", "ts_type"]
    assert out["unified_group_validate_strict"] is True
    assert out["auto_cls_model"] == "AutoNHITS"
    assert out["auto_h"] == 14
    assert out["auto_config_json"]["backend"] == "optuna"
    assert out["model_params_json"]["num_samples"] == 20
    assert out["param_space_json"]["seed"] == [1, 2]
    assert out["auto_callbacks_json"] == ["pkg.cb1", "pkg.cb2"]
    assert out["auto_num_samples"] == 11
    assert out["run_save"] is True
    assert out["run_load"] is False
    assert out["run_analyze"] is True
    assert out["run_predict"] is False
    assert out["recursive_depth"] == 1


def test_validate_meta_model_arguments_rejects_unknown_and_bad_types(monkeypatch):
    class DummyAdapter:
        def validate(self, model_name: str, model_params: dict):
            accepted = ["backend", "num_samples", "seed"]
            errors = []
            for k in model_params:
                if k not in accepted:
                    errors.append(f"unknown model params for {model_name}: ['{k}']")
            if "num_samples" in model_params and not isinstance(model_params["num_samples"], int):
                errors.append("num_samples must be int > 0")
            return {
                "ok": len(errors) == 0,
                "errors": errors,
                "warnings": [],
                "accepted_params": accepted,
                "required_model_params": [],
                "reserved_param_specs": {},
            }

    monkeypatch.setattr(
        "loto_forecast.orchestration.meta_automodel.get_adapter",
        lambda _name: DummyAdapter(),
    )

    report = _validate_meta_model_arguments(
        {
            "model_name": "AutoNHITS",
            "model_params_json": {"num_samples": "20", "hist_exog_list": ["x"]},
            "auto_config_json": {"backend": "optuna"},
            "param_space_json": {"bad_key": [1, 2]},
        }
    )
    assert report["ok"] is False
    assert any("model_params_json:" in e for e in report["errors"])
    assert any("param_space_json: unknown param key=bad_key" in e for e in report["errors"])
