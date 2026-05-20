from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Protocol

from ..orchestration.pipeline import evaluate, predict, train


def _auto_model_names() -> list[str]:
    from .neuralforecast_model import AUTO_MODEL_NAMES

    return list(AUTO_MODEL_NAMES)


def _resolve_model_class(model_name: str):
    from .neuralforecast_model import _resolve_model_class as resolver

    return resolver(model_name)


@dataclass
class AdapterRunResult:
    run_id: str
    train: dict[str, Any]
    predict: dict[str, Any] | None
    evaluate: dict[str, Any] | None


class ForecastAdapter(Protocol):
    name: str
    library_name: str

    def list_models(self) -> list[str]: ...

    def validate(self, model_name: str, model_params: dict[str, Any]) -> dict[str, Any]: ...

    def run(
        self,
        model_name: str,
        horizon: int,
        model_params: dict[str, Any],
        run_predict: bool,
        run_evaluate: bool,
        run_id: str,
        grid_id: str | None = None,
        task_id: int | None = None,
    ) -> AdapterRunResult: ...


class NeuralForecastAutoAdapter:
    name = "neuralforecast_auto"
    library_name = "neuralforecast"
    _reserved_specs: dict[str, dict[str, Any]] = {
        "backend": {"type": "str", "required": False, "allowed": ["optuna", "ray"]},
        "num_samples": {"type": "int>0", "required": False},
        "seed": {"type": "int", "required": False},
        "random_seed": {"type": "int", "required": False},
        "loss_name": {"type": "str", "required": False},
        "valid_loss_name": {"type": "str", "required": False},
        "search_alg_name": {"type": "str", "required": False},
        "cpus": {"type": "int>=0", "required": False},
        "gpus": {"type": "int>=0", "required": False},
        "refit_with_val": {"type": "bool", "required": False},
        "verbose": {"type": "bool", "required": False},
        "alias": {"type": "str", "required": False},
        "callbacks": {"type": "list", "required": False},
        "strict_exog": {"type": "bool", "required": False},
        "futr_exog_list": {"type": "list", "required": False},
        "hist_exog_list": {"type": "list", "required": False},
        "stat_exog_list": {"type": "list", "required": False},
        "run_cross_validation": {"type": "bool", "required": False},
        "local_scaler_type": {"type": "str", "required": False},
        "local_static_scaler_type": {"type": "str", "required": False},
        "freq": {"type": "str", "required": False},
        "dataset_input_method": {"type": "str", "required": False},
        "dataset_path": {"type": "str", "required": False},
        "dataset_sql": {"type": "str", "required": False},
        "dataframe_backend": {"type": "str", "required": False},
        "nf_fit_kwargs": {"type": "dict", "required": False},
        "nf_predict_kwargs": {"type": "dict", "required": False},
        "nf_cross_validation_kwargs": {"type": "dict", "required": False},
        "nf_save_kwargs": {"type": "dict", "required": False},
        "nf_load_kwargs": {"type": "dict", "required": False},
        "nf_predict_insample_kwargs": {"type": "dict", "required": False},
    }
    _reserved = set(_reserved_specs.keys())

    def list_models(self) -> list[str]:
        return sorted(_auto_model_names())

    def validate(self, model_name: str, model_params: dict[str, Any]) -> dict[str, Any]:
        from .neuralforecast_model import (
            get_model_exog_support,
            model_exog_support_table,
            validate_runtime_kwargs,
        )

        errors: list[str] = []
        warnings: list[str] = []

        if model_name not in _auto_model_names():
            errors.append(f"unsupported model_name={model_name}. choices={self.list_models()}")
            return {"ok": False, "errors": errors, "warnings": warnings}

        try:
            cls = _resolve_model_class(model_name)
        except Exception as e:
            errors.append(str(e))
            return {"ok": False, "errors": errors, "warnings": warnings}
        sig = inspect.signature(cls.__init__)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        known = {
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        required_model_params = {
            p.name
            for p in sig.parameters.values()
            if p.name != "self"
            and p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            and p.default is inspect._empty
            and p.name not in {"h", "loss", "valid_loss"}
        }

        unknown = [k for k in model_params if k not in self._reserved and k not in known]
        if unknown and not accepts_var_kw:
            errors.append(f"unknown model params for {model_name}: {unknown}")

        for req_key in sorted(required_model_params):
            if req_key not in model_params:
                errors.append(f"missing required model param for {model_name}: {req_key}")

        for key, value in model_params.items():
            if key not in self._reserved_specs:
                continue
            spec = self._reserved_specs[key]
            typ = str(spec.get("type") or "").strip()
            if typ == "str":
                if not isinstance(value, str):
                    errors.append(f"{key} must be str")
                elif key == "backend":
                    allowed = set(spec.get("allowed") or [])
                    if value not in allowed:
                        errors.append(f"{key} must be one of {sorted(allowed)}")
            elif typ == "int":
                if not isinstance(value, int):
                    errors.append(f"{key} must be int")
            elif typ == "int>0":
                if not isinstance(value, int) or int(value) <= 0:
                    errors.append(f"{key} must be int > 0")
            elif typ == "int>=0":
                if not isinstance(value, int) or int(value) < 0:
                    errors.append(f"{key} must be int >= 0")
            elif typ == "bool":
                if not isinstance(value, bool):
                    errors.append(f"{key} must be bool")
            elif typ == "list":
                if not isinstance(value, list):
                    errors.append(f"{key} must be list")
            elif typ == "dict" and not isinstance(value, dict):
                errors.append(f"{key} must be dict")

        loss_name = model_params.get("loss_name")
        valid_loss_name = model_params.get("valid_loss_name")
        if (
            isinstance(loss_name, str)
            and isinstance(valid_loss_name, str)
            and valid_loss_name.strip()
            and valid_loss_name.strip().upper() != loss_name.strip().upper()
        ):
            errors.append("valid_loss_name must match loss_name")
        scaler_a = model_params.get("local_scaler_type")
        scaler_b = model_params.get("local_static_scaler_type")
        if (
            isinstance(scaler_a, str)
            and isinstance(scaler_b, str)
            and scaler_a.strip()
            and scaler_b.strip()
            and scaler_a.strip() != scaler_b.strip()
        ):
            errors.append("local_static_scaler_type must match local_scaler_type")

        if str(model_name) == "AutoHINT":
            backend_val = str(model_params.get("backend", "ray")).strip().lower()
            if backend_val == "optuna":
                errors.append("AutoHINT does not support backend=optuna. Use backend=ray.")
            cfg_val = model_params.get("config")
            if isinstance(cfg_val, dict) and "reconciliation" not in cfg_val:
                errors.append("AutoHINT config must include 'reconciliation'.")

        exog_support = get_model_exog_support(model_name)
        exog_role_map = {
            "futr_exog_list": "futr",
            "hist_exog_list": "hist",
            "stat_exog_list": "stat",
        }
        for key, role in exog_role_map.items():
            vals = model_params.get(key, [])
            if not isinstance(vals, list):
                continue
            cols = [str(v).strip() for v in vals if str(v).strip()]
            if cols and not exog_support.get(role, False):
                errors.append(f"{model_name} does not support {key} but received {cols[:10]}")

        runtime_keys = {
            "nf_fit_kwargs",
            "nf_predict_kwargs",
            "nf_cross_validation_kwargs",
            "nf_save_kwargs",
            "nf_load_kwargs",
            "nf_predict_insample_kwargs",
        }
        runtime_in = {k: model_params.get(k) for k in runtime_keys if k in model_params}
        runtime_report = validate_runtime_kwargs(runtime_in)
        if not runtime_report.get("ok", False):
            errors.extend(runtime_report.get("errors", []))
        warnings.extend(runtime_report.get("warnings", []))

        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "accepted_params": sorted(list(known | self._reserved)),
            "required_model_params": sorted(list(required_model_params)),
            "unknown_params": sorted(unknown),
            "reserved_param_specs": self._reserved_specs,
            "model_exog_support": exog_support,
            "model_exog_support_table": model_exog_support_table(),
        }

    def run(
        self,
        model_name: str,
        horizon: int,
        model_params: dict[str, Any],
        run_predict: bool,
        run_evaluate: bool,
        run_id: str,
        grid_id: str | None = None,
        task_id: int | None = None,
    ) -> AdapterRunResult:
        train_out = train(
            model_name=model_name,
            h=horizon,
            model_params=model_params,
            run_id=run_id,
            library_name=self.library_name,
            adapter_name=self.name,
            grid_id=grid_id,
            task_id=task_id,
        )

        predict_out: dict[str, Any] | None = None
        eval_out: dict[str, Any] | None = None

        if run_predict:
            fcst = predict(run_id=train_out["run_id"], h=horizon)
            predict_out = {
                "n_rows": int(len(fcst)),
                "columns": list(fcst.columns),
            }

        if run_evaluate:
            eval_out = evaluate(run_id=train_out["run_id"])

        return AdapterRunResult(
            run_id=train_out["run_id"],
            train=train_out,
            predict=predict_out,
            evaluate=eval_out,
        )


_ADAPTERS: dict[str, ForecastAdapter] = {
    NeuralForecastAutoAdapter.name: NeuralForecastAutoAdapter(),
}


def register_adapter(name: str, adapter: ForecastAdapter) -> None:
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> ForecastAdapter:
    if name not in _ADAPTERS:
        raise ValueError(f"adapter not found: {name}. available={list(_ADAPTERS)}")
    return _ADAPTERS[name]


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS.keys())
