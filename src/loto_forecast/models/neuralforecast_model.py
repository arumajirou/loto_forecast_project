from __future__ import annotations

import copy
import importlib
import inspect
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from pandas.errors import PerformanceWarning

from ..config.settings import settings
from ..features.engineering import infer_exog_columns

_FALLBACK_AUTO_MODEL_NAMES = [
    "AutoAutoformer",
    "AutoBiTCN",
    "AutoDLinear",
    "AutoDeepAR",
    "AutoDeepNPTS",
    "AutoDilatedRNN",
    "AutoFEDformer",
    "AutoGRU",
    "AutoHINT",
    "AutoInformer",
    "AutoiTransformer",
    "AutoKAN",
    "AutoLSTM",
    "AutoMLP",
    "AutoMLPMultivariate",
    "AutoNBEATS",
    "AutoNBEATSx",
    "AutoNHITS",
    "AutoNLinear",
    "AutoPatchTST",
    "AutoRMoK",
    "AutoRNN",
    "AutoSOFTS",
    "AutoStemGNN",
    "AutoTCN",
    "AutoTFT",
    "AutoTiDE",
    "AutoTimeMixer",
    "AutoTimesNet",
    "AutoTimeXer",
    "AutoTSMixer",
    "AutoTSMixerx",
    "AutoVanillaTransformer",
    "AutoxLSTM",
]


def _discover_auto_model_names() -> list[str]:
    try:
        nf_auto = importlib.import_module("neuralforecast.auto")
        names: list[str] = []
        for name in dir(nf_auto):
            if not str(name).startswith("Auto"):
                continue
            obj = getattr(nf_auto, name, None)
            if inspect.isclass(obj):
                names.append(str(name))
        if names:
            return sorted(list(dict.fromkeys(names)))
    except Exception:
        pass
    return sorted(list(dict.fromkeys(_FALLBACK_AUTO_MODEL_NAMES)))


AUTO_MODEL_NAMES = _discover_auto_model_names()

MODEL_EXOG_SUPPORT: dict[str, dict[str, bool]] = {
    # F: future exogenous, H: historic exogenous, S: static exogenous
    "AutoAutoformer": {"futr": True, "hist": False, "stat": False},
    "AutoBiTCN": {"futr": True, "hist": True, "stat": True},
    "AutoDLinear": {"futr": False, "hist": False, "stat": False},
    "AutoDeepAR": {"futr": True, "hist": False, "stat": True},
    "AutoDeepNPTS": {"futr": True, "hist": True, "stat": True},
    "AutoDilatedRNN": {"futr": True, "hist": True, "stat": True},
    "AutoFEDformer": {"futr": True, "hist": False, "stat": False},
    "AutoGRU": {"futr": True, "hist": True, "stat": True},
    "AutoHINT": {"futr": True, "hist": True, "stat": True},
    "AutoInformer": {"futr": True, "hist": False, "stat": False},
    "AutoiTransformer": {"futr": False, "hist": False, "stat": False},
    "AutoKAN": {"futr": True, "hist": True, "stat": True},
    "AutoLSTM": {"futr": True, "hist": True, "stat": True},
    "AutoMLP": {"futr": True, "hist": True, "stat": True},
    "AutoMLPMultivariate": {"futr": True, "hist": True, "stat": True},
    "AutoNBEATS": {"futr": False, "hist": False, "stat": False},
    "AutoNHITS": {"futr": True, "hist": True, "stat": True},
    "AutoNBEATSx": {"futr": True, "hist": True, "stat": True},
    "AutoNLinear": {"futr": False, "hist": False, "stat": False},
    "AutoPatchTST": {"futr": False, "hist": False, "stat": False},
    "AutoRMoK": {"futr": False, "hist": False, "stat": False},
    "AutoRNN": {"futr": True, "hist": True, "stat": True},
    "AutoSOFTS": {"futr": False, "hist": False, "stat": False},
    "AutoStemGNN": {"futr": False, "hist": False, "stat": False},
    "AutoTCN": {"futr": True, "hist": True, "stat": True},
    "AutoTFT": {"futr": True, "hist": True, "stat": True},
    "AutoTiDE": {"futr": True, "hist": True, "stat": True},
    "AutoTimeMixer": {"futr": False, "hist": False, "stat": False},
    "AutoTimesNet": {"futr": True, "hist": False, "stat": False},
    "AutoTimeXer": {"futr": True, "hist": False, "stat": False},
    "AutoTSMixer": {"futr": False, "hist": False, "stat": False},
    "AutoTSMixerx": {"futr": True, "hist": True, "stat": True},
    "AutoVanillaTransformer": {"futr": True, "hist": False, "stat": False},
    "AutoxLSTM": {"futr": True, "hist": True, "stat": True},
}

H1_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    # Avoid h=1 incompatible defaults in underlying NeuralForecast models.
    "AutoNBEATS": {"stack_types": ["identity", "identity", "identity"]},
    "AutoNBEATSx": {"stack_types": ["identity", "identity", "identity"]},
    "AutoAutoformer": {"input_size": 2},
    "AutoInformer": {"input_size": 2},
    "AutoVanillaTransformer": {"input_size": 2},
    "AutoTimeMixer": {"input_size": 2},
    "AutoTimesNet": {"top_k": 1},
    "AutoTimeXer": {"patch_len": 1},
}
SMALL_H_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    # Autoformer-based autocorrelation uses top_k=int(factor*log(length)).
    # For very short horizons, factor=3 can exceed available correlation length.
    "AutoAutoformer": {"factor": 1},
}
MODEL_MIN_H_RULES: dict[str, int] = {
    "AutoNBEATS": 2,
    "AutoNBEATSx": 2,
    "AutoAutoformer": 2,
    "Autoformer": 2,
    "AutoInformer": 2,
    "AutoFEDformer": 2,
    "AutoVanillaTransformer": 2,
    "AutoTimeMixer": 2,
    "AutoTimesNet": 2,
    "AutoTimeXer": 2,
    "AutoPatchTST": 2,
    "AutoRMoK": 2,
    "AutoStemGNN": 2,
}

NF_RUNTIME_KWARG_SPECS: dict[str, dict[str, Any]] = {
    "nf_fit_kwargs": {
        "allowed": {
            "static_df",
            "val_size",
            "use_init_models",
            "verbose",
            "id_col",
            "time_col",
            "target_col",
            "distributed_config",
            "prediction_intervals",
        },
        "blocked": {"df"},
    },
    "nf_predict_kwargs": {
        "allowed": {
            "static_df",
            "futr_df",
            "verbose",
            "engine",
            "level",
            "quantiles",
            "h",
            "data_kwargs",
        },
        "blocked": {"df"},
    },
    "nf_cross_validation_kwargs": {
        "allowed": {
            "static_df",
            "n_windows",
            "step_size",
            "val_size",
            "test_size",
            "use_init_models",
            "verbose",
            "refit",
            "id_col",
            "time_col",
            "target_col",
            "prediction_intervals",
            "level",
            "quantiles",
            "h",
            "data_kwargs",
        },
        "blocked": {"df"},
    },
    "nf_save_kwargs": {
        "allowed": {"model_index", "save_dataset", "overwrite"},
        "blocked": {"path"},
    },
    "nf_load_kwargs": {
        "allowed": {"verbose"},
        "allow_any": True,
        "blocked": {"path"},
    },
    "nf_predict_insample_kwargs": {
        "allowed": {"step_size", "level", "quantiles"},
        "blocked": set(),
    },
}

MAX_EXOG_COLS_PER_ROLE = 256

# utilsforecast internals may emit fragmentation warnings for wide frames.
warnings.filterwarnings("ignore", category=PerformanceWarning, module=r"utilsforecast\.processing")


def _normalize_exog_dict(exog: dict[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"futr_exog": [], "hist_exog": [], "stat_exog": []}
    source = dict(exog or {})
    for role in ("futr_exog", "hist_exog", "stat_exog"):
        cols = _json_list(source.get(role))
        out[role] = list(dict.fromkeys([str(c).strip() for c in cols if str(c).strip()]))
    return out


def _apply_model_exog_support(
    model_name: str,
    exog: dict[str, list[str]],
    strict_exog: bool,
    explicit_roles: set[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    explicit_roles = set(explicit_roles or set())
    support = get_model_exog_support(model_name)
    role_map = {
        "futr_exog": "futr",
        "hist_exog": "hist",
        "stat_exog": "stat",
    }
    out = {k: list(v) for k, v in exog.items()}
    warnings_out: list[str] = []
    for role, support_key in role_map.items():
        values = list(out.get(role, []))
        if not values:
            continue
        if support.get(support_key, False):
            continue
        msg = f"{model_name} does not support {role} but received {len(values)} column(s): {values[:10]}"
        if role in explicit_roles and strict_exog:
            raise ValueError(msg)
        warnings_out.append(msg)
        out[role] = []
    return out, warnings_out


def _configure_torch_runtime() -> None:
    """Apply safe runtime tweaks when CUDA is available."""
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
    except Exception:
        # Keep training/prediction robust even if runtime tuning is unsupported.
        return


def _resolve_model_class(model_name: str):
    if model_name not in AUTO_MODEL_NAMES:
        raise ValueError(f"Unsupported model_name={model_name}. choices={AUTO_MODEL_NAMES}")
    try:
        nf_auto = importlib.import_module("neuralforecast.auto")
    except Exception as e:
        raise RuntimeError(
            "Failed to import neuralforecast.auto. "
            "Install compatible versions of neuralforecast/pytorch-lightning/setuptools. "
            f"detail={e}"
        ) from e
    cls = getattr(nf_auto, str(model_name), None)
    if cls is None or not inspect.isclass(cls):
        available = sorted(
            [str(n) for n in dir(nf_auto) if str(n).startswith("Auto") and inspect.isclass(getattr(nf_auto, n, None))]
        )
        raise ValueError(f"model class not found: {model_name}. available={available[:80]}")
    return cls


def _load_neuralforecast_runtime():
    from ..patches.neuralforecast_autoformer_safe_topk import apply as apply_autoformer_safe_topk

    apply_autoformer_safe_topk()
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE
    except Exception as e:
        raise RuntimeError(
            "Failed to import neuralforecast runtime. "
            "Install compatible versions of neuralforecast/pytorch-lightning/setuptools. "
            f"detail={e}"
        ) from e
    return NeuralForecast, MAE


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y"}:
        return True
    if raw in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _merge_model_config_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return config
    if callable(config):
        base_fn = config
        patch = dict(overrides)

        def _wrapped(trial):
            base = base_fn(trial)
            out = dict(base) if isinstance(base, dict) else {}
            out.update(patch)
            return out

        return _wrapped
    if isinstance(config, dict):
        out = dict(config)
        out.update(dict(overrides))
        return out
    return None


def _normalize_auto_config_for_backend(config: Any, backend: str, model_cls: Any) -> Any:
    if str(backend).strip().lower() == "optuna" and isinstance(config, dict):
        converter = getattr(model_cls, "_ray_config_to_optuna", None)
        if callable(converter):
            try:
                return converter(config)
            except Exception:
                pass

        def _constant_optuna_config(_trial):
            return dict(config)

        return _constant_optuna_config
    return config


def _default_auto_config(
    model_cls: Any,
    h: int,
    backend: str,
    n_series: Any | None = None,
) -> Any:
    get_default = getattr(model_cls, "get_default_config", None)
    if not callable(get_default):
        return None
    try:
        sig = inspect.signature(get_default)
        if "n_series" in sig.parameters:
            resolved_n_series = max(1, _safe_int(n_series, 1))
            return get_default(h=h, backend=backend, n_series=resolved_n_series)
        return get_default(h=h, backend=backend)
    except Exception:
        return None


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            loaded = json.loads(raw)
            return list(loaded) if isinstance(loaded, list) else []
        except Exception:
            return []
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            loaded = json.loads(raw)
            return dict(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


def _build_loss(loss_name: str | None):
    _, MAE = _load_neuralforecast_runtime()
    if not loss_name:
        return MAE()
    name = str(loss_name).strip().upper()
    try:
        from neuralforecast.losses import pytorch as nfloss
    except Exception:
        return MAE()
    mapping = {
        "MAE": "MAE",
        "MSE": "MSE",
        "RMSE": "RMSE",
        "MAPE": "MAPE",
        "SMAPE": "SMAPE",
        "HUBERLOSS": "HuberLoss",
        "HUBER": "HuberLoss",
    }
    cls_name = mapping.get(name)
    if not cls_name or not hasattr(nfloss, cls_name):
        return MAE()
    try:
        return getattr(nfloss, cls_name)()
    except Exception:
        return MAE()


def _build_search_alg(search_alg_name: str | None, backend: str, seed: int):
    if not search_alg_name:
        return None
    key = str(search_alg_name).strip()
    if not key:
        return None
    low = key.lower()
    if backend == "optuna":
        try:
            import optuna
        except Exception:
            return None
        if low in {"randomsampler", "random"}:
            return optuna.samplers.RandomSampler(seed=seed)
        if low in {"tpesampler", "tpe"}:
            return optuna.samplers.TPESampler(seed=seed)
        if low in {"cmaessampler", "cmaes"}:
            try:
                return optuna.samplers.CmaEsSampler(seed=seed)
            except Exception:
                return None
        if low in {"nsgaiisampler", "nsgaii"}:
            try:
                return optuna.samplers.NSGAIISampler(seed=seed)
            except Exception:
                return None
        return None
    if low in {"basicvariantgenerator", "basic", "default"}:
        try:
            from ray.tune.search.basic_variant import BasicVariantGenerator
        except Exception:
            return None
        return BasicVariantGenerator(random_state=seed)
    if low in {"optunasearch", "optuna"}:
        try:
            from ray.tune.search.optuna import OptunaSearch
        except Exception:
            return None
        return OptunaSearch()
    if low in {"hyperoptsearch", "hyperopt"}:
        try:
            from ray.tune.search.hyperopt import HyperOptSearch
        except Exception:
            return None
        return HyperOptSearch(random_state_seed=seed)
    if low in {"bayesoptsearch", "bayesopt"}:
        try:
            from ray.tune.search.bayesopt import BayesOptSearch
        except Exception:
            return None
        return BayesOptSearch()
    return None


def _load_callbacks(callback_paths: list[Any]) -> list[Any] | None:
    callbacks: list[Any] = []
    for raw in callback_paths:
        if not isinstance(raw, str):
            continue
        path = raw.strip()
        if not path or "." not in path:
            continue
        mod_name, attr_name = path.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_name)
            cb = getattr(mod, attr_name)
            callbacks.append(cb)
        except Exception as e:
            logger.warning(f"callback load failed path={path}: {e}")
    return callbacks or None


def get_model_exog_support(model_name: str) -> dict[str, bool]:
    support = MODEL_EXOG_SUPPORT.get(str(model_name), {"futr": False, "hist": False, "stat": False})
    return {
        "futr": bool(support.get("futr", False)),
        "hist": bool(support.get("hist", False)),
        "stat": bool(support.get("stat", False)),
    }


def model_exog_support_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(AUTO_MODEL_NAMES):
        support = get_model_exog_support(name)
        rows.append(
            {
                "model_name": name,
                "supports_futr_exog": bool(support["futr"]),
                "supports_hist_exog": bool(support["hist"]),
                "supports_stat_exog": bool(support["stat"]),
                "exogenous_code": "".join(
                    [
                        "F" if support["futr"] else "-",
                        "H" if support["hist"] else "-",
                        "S" if support["stat"] else "-",
                    ]
                ),
            }
        )
    return rows


def validate_runtime_kwargs(raw: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(raw or {})
    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    for key, spec in NF_RUNTIME_KWARG_SPECS.items():
        if key not in params or params.get(key) is None:
            normalized[key] = {}
            continue
        value = params.get(key)
        if not isinstance(value, dict):
            errors.append(f"{key} must be dict")
            normalized[key] = {}
            continue
        allowed = set(spec.get("allowed") or [])
        blocked = set(spec.get("blocked") or [])
        allow_any = bool(spec.get("allow_any", False))
        clean: dict[str, Any] = {}
        for k, v in value.items():
            name = str(k)
            if name in blocked:
                errors.append(f"{key}.{name} is not allowed")
                continue
            if (allowed and name not in allowed) and not allow_any:
                errors.append(f"{key}: unknown option '{name}'")
                continue
            clean[name] = v
        normalized[key] = clean
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "normalized": normalized}


@dataclass
class TrainResult:
    run_id: str
    model_name: str
    artifact_path: Path
    exog: dict


def prepare_nf_frames(
    df: pd.DataFrame,
    exog: dict[str, list[str]] | None = None,
    futr_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, list[str]]]:
    """Prepare model-ready frames with safe exogenous columns."""
    required = [settings.id_col, settings.time_col, settings.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    base = df.copy()
    base[settings.time_col] = pd.to_datetime(base[settings.time_col], errors="coerce")
    base = base.dropna(subset=required).sort_values([settings.id_col, settings.time_col]).reset_index(drop=True)
    if base.empty:
        raise ValueError("prepare_nf_frames: empty dataframe after required-column filtering")

    exog = exog or infer_exog_columns(base)
    clean_exog: dict[str, list[str]] = {}
    for role in ("futr_exog", "hist_exog", "stat_exog"):
        cols: list[str] = []
        for c in exog.get(role, []) or []:
            if c not in base.columns:
                continue
            if not (pd.api.types.is_numeric_dtype(base[c]) or pd.api.types.is_bool_dtype(base[c])):
                continue
            cols.append(str(c))
        clean_exog[role] = list(dict.fromkeys(cols))

    used_exog = clean_exog["futr_exog"] + clean_exog["hist_exog"] + clean_exog["stat_exog"]
    fit_cols = required + [c for c in used_exog if c not in required]
    fit_df = base[fit_cols].copy()

    fill_values: dict[str, float] = {}
    for c in used_exog:
        fit_df[c] = pd.to_numeric(fit_df[c], errors="coerce")
        median = fit_df[c].median(skipna=True)
        fill = 0.0 if pd.isna(median) else float(median)
        fill_values[c] = fill
        fit_df[c] = fit_df[c].fillna(fill)

    # De-fragment frame before handing over to utilsforecast/neuralforecast.
    fit_df = fit_df.copy()

    futr_out: pd.DataFrame | None = None
    if futr_df is not None:
        futr_need = [settings.id_col, settings.time_col]
        futr_missing = [c for c in futr_need if c not in futr_df.columns]
        if futr_missing:
            raise ValueError(f"futr_df missing required columns: {futr_missing}")
        futr_out = futr_df.copy()
        futr_out[settings.time_col] = pd.to_datetime(futr_out[settings.time_col], errors="coerce")
        futr_out = (
            futr_out.dropna(subset=futr_need).sort_values([settings.id_col, settings.time_col]).reset_index(drop=True)
        )
        for c in clean_exog["futr_exog"]:
            if c not in futr_out.columns:
                futr_out[c] = fill_values.get(c, 0.0)
            futr_out[c] = pd.to_numeric(futr_out[c], errors="coerce").fillna(fill_values.get(c, 0.0))
        futr_out = futr_out[futr_need + clean_exog["futr_exog"]].copy()

    return fit_df, futr_out, clean_exog


def _resolve_nf_fit_kwargs_for_runtime(fit_df: pd.DataFrame, fit_kwargs_in: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(fit_kwargs_in or {})
    static_raw = out.get("static_df")
    if static_raw is None:
        return out

    def _parse_static_cols(raw: Any) -> list[str]:
        if isinstance(raw, list):
            return [str(c).strip() for c in raw if str(c).strip()]
        if isinstance(raw, dict):
            mode = str(raw.get("mode", "")).strip().lower()
            cols = raw.get("columns")
            if isinstance(cols, list):
                parsed = [str(c).strip() for c in cols if str(c).strip()]
                if parsed:
                    return parsed
            if mode in {"auto", "stat_auto", "stat_columns"}:
                return [str(c) for c in fit_df.columns if str(c).startswith("stat_")]
            return []
        if isinstance(raw, str):
            txt = raw.strip()
            if not txt:
                return []
            if txt.lower() in {"(none)", "none", "null"}:
                return []
            if txt.lower() in {"auto", "stat_auto", "stat_columns"}:
                return [str(c) for c in fit_df.columns if str(c).startswith("stat_")]
            try:
                loaded = json.loads(txt)
                return _parse_static_cols(loaded)
            except Exception:
                return [x.strip() for x in txt.split(",") if x.strip()]
        return []

    id_col = str(out.get("id_col") or settings.id_col)
    if id_col not in fit_df.columns:
        if settings.id_col in fit_df.columns:
            id_col = str(settings.id_col)
        else:
            out.pop("static_df", None)
            return out

    cols = [c for c in _parse_static_cols(static_raw) if c in fit_df.columns and c != id_col]
    cols = list(dict.fromkeys(cols))
    if not cols:
        out.pop("static_df", None)
        return out

    out["static_df"] = fit_df[[id_col] + cols].drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
    return out


def build_automodel(
    model_name: str,
    h: int,
    exog: dict,
    backend: str = "optuna",
    num_samples: int = 25,
    seed: int = 1,
    loss_name: str = "MAE",
    valid_loss_name: str | None = None,
    search_alg_name: str | None = None,
    cpus: int | None = None,
    gpus: int | None = None,
    refit_with_val: bool = False,
    verbose: bool = False,
    alias: str | None = None,
    callbacks: list[Any] | None = None,
    model_kwargs: dict | None = None,
):
    """Build NeuralForecast AutoModel with extensible kwargs."""
    _load_neuralforecast_runtime()
    ModelCls = _resolve_model_class(model_name)
    sig = inspect.signature(ModelCls.__init__)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    accepted_keys = {
        p.name
        for p in sig.parameters.values()
        if p.name != "self" and p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }

    resolved_loss_name = str(loss_name or "MAE")
    resolved_valid_loss_name = resolved_loss_name
    if valid_loss_name is not None and str(valid_loss_name).strip():
        provided_valid = str(valid_loss_name).strip()
        if provided_valid.upper() != resolved_loss_name.strip().upper():
            logger.warning(
                f"valid_loss_name({provided_valid}) ignored; forcing valid_loss_name=loss_name({resolved_loss_name})"
            )

    kwargs = dict(
        h=h,
        loss=_build_loss(resolved_loss_name),
    )
    # Pass tuning/runtime args only when the model class explicitly declares them.
    # Some classes accept **kwargs but forward unknown keys to lower-level trainers.
    if "num_samples" in accepted_keys:
        kwargs["num_samples"] = int(num_samples)
    if "backend" in accepted_keys:
        kwargs["backend"] = str(backend)
    valid_loss_param = sig.parameters.get("valid_loss")
    valid_loss_required = bool(
        valid_loss_param is not None
        and valid_loss_param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and valid_loss_param.default is inspect._empty
    )
    if accepts_var_kw or "valid_loss" in accepted_keys or valid_loss_required:
        kwargs["valid_loss"] = _build_loss(resolved_valid_loss_name)

    search_alg = _build_search_alg(search_alg_name, backend=backend, seed=seed)
    if search_alg is not None and "search_alg" in accepted_keys:
        kwargs["search_alg"] = search_alg
    if cpus is not None and "cpus" in accepted_keys:
        kwargs["cpus"] = int(cpus)
    if gpus is not None and "gpus" in accepted_keys:
        kwargs["gpus"] = int(gpus)
    if refit_with_val and "refit_with_val" in accepted_keys:
        kwargs["refit_with_val"] = True
    if verbose and "verbose" in accepted_keys:
        kwargs["verbose"] = True
    if alias and "alias" in accepted_keys:
        kwargs["alias"] = str(alias)
    if callbacks and "callbacks" in accepted_keys:
        kwargs["callbacks"] = callbacks

    if exog.get("hist_exog") and ("hist_exog_list" in accepted_keys):
        kwargs["hist_exog_list"] = exog["hist_exog"]
    if exog.get("futr_exog") and ("futr_exog_list" in accepted_keys):
        kwargs["futr_exog_list"] = exog["futr_exog"]
    if exog.get("stat_exog") and ("stat_exog_list" in accepted_keys):
        kwargs["stat_exog_list"] = exog["stat_exog"]

    if model_kwargs:
        kwargs.update(model_kwargs)

    h_int = _safe_int(h, 1)
    compat_notes: list[str] = []
    config_patch: dict[str, Any] = {}
    if h_int == 1:
        config_patch.update(dict(H1_CONFIG_OVERRIDES.get(str(model_name), {})))
    if h_int <= 2:
        config_patch.update(dict(SMALL_H_CONFIG_OVERRIDES.get(str(model_name), {})))
    if config_patch and (accepts_var_kw or "config" in accepted_keys):
        cfg = kwargs.get("config")
        if cfg is None:
            cfg = _default_auto_config(
                model_cls=ModelCls,
                h=h_int,
                backend=str(backend),
                n_series=kwargs.get("n_series"),
            )
        cfg = _normalize_auto_config_for_backend(cfg, backend=str(backend), model_cls=ModelCls)
        merged = _merge_model_config_overrides(cfg, config_patch)
        if merged is None:
            if str(backend).strip().lower() == "optuna":
                local_patch = dict(config_patch)

                def _constant_optuna_config(_trial):
                    return dict(local_patch)

                merged = _constant_optuna_config
            else:
                merged = dict(config_patch)
        kwargs["config"] = merged
        compat_notes.append(f"config_overrides={config_patch}")

    # Autoformer (not BaseAuto) requires input_size > 1 when h=1 and default decoder multiplier is used.
    if h_int == 1 and str(model_name) == "Autoformer" and (accepts_var_kw or "input_size" in accepted_keys):
        input_size_v = max(1, _safe_int(kwargs.get("input_size"), 1))
        if input_size_v < 2:
            kwargs["input_size"] = 2
            compat_notes.append("input_size=2")

    if compat_notes:
        logger.warning(
            f"small-h compatibility override applied model={model_name} h={int(h_int)}: " + ", ".join(compat_notes)
        )

    if not accepts_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in accepted_keys}

    return ModelCls(**kwargs)


def train_automodel(
    df: pd.DataFrame,
    model_name: str,
    h: int | None = None,
    freq: str | None = None,
    run_id: str | None = None,
    model_params: dict | None = None,
) -> TrainResult:
    h = h or settings.default_horizon
    min_h_required = int(MODEL_MIN_H_RULES.get(str(model_name), 1) or 1)
    if int(h) < min_h_required:
        logger.warning(f"h={int(h)} is too small for {model_name}; auto-adjust to h={int(min_h_required)}")
        h = int(min_h_required)
    freq = freq or settings.freq
    run_id = run_id or f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    if df.empty:
        raise ValueError("train_automodel received empty dataframe")
    missing_required = {settings.id_col, settings.time_col, settings.target_col} - set(df.columns)
    if missing_required:
        raise ValueError(f"train_automodel missing required columns: {sorted(missing_required)}")

    model_params = copy.deepcopy(model_params) if model_params else {}
    model_params = {str(k): v for k, v in model_params.items() if not (isinstance(v, str) and str(v).strip() == "")}
    backend = str(model_params.pop("backend", "optuna"))
    num_samples = _safe_int(model_params.pop("num_samples", 25), 25)
    seed = _safe_int(model_params.pop("seed", 1), 1)
    if "random_seed" in model_params:
        seed = _safe_int(model_params.pop("random_seed"), seed)
    loss_name = str(model_params.pop("loss_name", "MAE"))
    valid_loss_name = model_params.pop("valid_loss_name", None)
    search_alg_name = model_params.pop("search_alg_name", None)
    cpus = model_params.pop("cpus", None)
    gpus = model_params.pop("gpus", None)
    refit_with_val = _to_bool(model_params.pop("refit_with_val", False), False)
    verbose = _to_bool(model_params.pop("verbose", False), False)
    alias = model_params.pop("alias", None)
    callbacks_raw = model_params.pop("callbacks", None)
    callbacks = _load_callbacks(_json_list(callbacks_raw))
    strict_exog = _to_bool(model_params.pop("strict_exog", False), False)
    run_cross_validation = _to_bool(model_params.pop("run_cross_validation", False), False)
    local_scaler_type = model_params.pop("local_scaler_type", None)
    local_static_scaler_type = model_params.pop("local_static_scaler_type", None)

    def _normalize_scaler_name(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() in {"none", "(none)", "null"}:
            return None
        return text

    local_scaler_type = _normalize_scaler_name(local_scaler_type)
    local_static_scaler_type = _normalize_scaler_name(local_static_scaler_type)
    if local_scaler_type and local_static_scaler_type and local_scaler_type != local_static_scaler_type:
        logger.warning("local_scaler_type and local_static_scaler_type differ; forcing both to local_scaler_type value")
    resolved_local_scaler_type = local_scaler_type or local_static_scaler_type
    local_scaler_type = resolved_local_scaler_type
    local_static_scaler_type = resolved_local_scaler_type

    if valid_loss_name is not None and str(valid_loss_name).strip():
        provided_valid_loss = str(valid_loss_name).strip()
        if provided_valid_loss.upper() != str(loss_name).strip().upper():
            logger.warning(
                f"valid_loss_name({provided_valid_loss}) ignored; forcing valid_loss_name=loss_name({loss_name})"
            )
    valid_loss_name = str(loss_name)

    for core_key in ("h", "loss", "valid_loss", "search_alg"):
        if core_key in model_params:
            logger.warning(f"ignore unsupported core override key in params_json: {core_key}")
            model_params.pop(core_key, None)

    required_model_params: set[str] = set()
    try:
        cls_for_req = _resolve_model_class(model_name)
        sig_for_req = inspect.signature(cls_for_req.__init__)
        required_model_params = {
            p.name
            for p in sig_for_req.parameters.values()
            if p.name != "self"
            and p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            and p.default is inspect._empty
            and p.name not in {"h", "loss", "valid_loss"}
        }
    except Exception:
        required_model_params = set()

    auto_n_series = max(1, int(df[settings.id_col].nunique()))
    if "n_series" in required_model_params:
        had_n_series = "n_series" in model_params
        resolved_n_series = _safe_int(model_params.get("n_series"), auto_n_series)
        if int(resolved_n_series) <= 0:
            resolved_n_series = int(auto_n_series)
        model_params["n_series"] = int(resolved_n_series)
        if not had_n_series:
            logger.info(f"auto-filled required model param n_series={int(resolved_n_series)}")

    if "input_size" in required_model_params and "input_size" not in model_params:
        # Prefer a small positive fallback tied to horizon instead of leaving required params empty.
        fallback_input_size = max(1, int(h))
        # Models with decoder_input_size_multiplier require input_size > 1 when h=1.
        if int(h) == 1 and "decoder_input_size_multiplier" in sig_for_req.parameters:
            fallback_input_size = max(2, fallback_input_size)
        model_params["input_size"] = int(fallback_input_size)
        logger.info(f"auto-filled required model param input_size={int(model_params['input_size'])}")

    explicit_exog_roles: set[str] = set()
    raw_exog = _normalize_exog_dict(infer_exog_columns(df))
    role_keys = {
        "futr_exog": ("futr_exog_list", "futr_exog"),
        "hist_exog": ("hist_exog_list", "hist_exog"),
        "stat_exog": ("stat_exog_list", "stat_exog"),
    }
    for role, aliases in role_keys.items():
        explicit_values: list[str] | None = None
        for alias_key in aliases:
            if alias_key in model_params:
                explicit_exog_roles.add(role)
                parsed = _json_list(model_params.pop(alias_key))
                explicit_values = (
                    parsed if explicit_values is None else list(dict.fromkeys([*explicit_values, *parsed]))
                )
        if explicit_values is not None:
            raw_exog[role] = list(dict.fromkeys([str(v).strip() for v in explicit_values if str(v).strip()]))
    exog_input, exog_warnings = _apply_model_exog_support(
        model_name=model_name,
        exog=raw_exog,
        strict_exog=strict_exog,
        explicit_roles=explicit_exog_roles,
    )
    for role in ("futr_exog", "hist_exog", "stat_exog"):
        cols_role = list(exog_input.get(role, []))
        if len(cols_role) > int(MAX_EXOG_COLS_PER_ROLE):
            logger.warning(
                f"{role} has {len(cols_role)} columns; cap to first {int(MAX_EXOG_COLS_PER_ROLE)} for runtime stability"
            )
            exog_input[role] = cols_role[: int(MAX_EXOG_COLS_PER_ROLE)]

    runtime_kwargs_raw = {key: _json_dict(model_params.pop(key, None)) for key in NF_RUNTIME_KWARG_SPECS}
    runtime_kwargs_validation = validate_runtime_kwargs(runtime_kwargs_raw)
    if not runtime_kwargs_validation["ok"]:
        raise ValueError(
            "invalid runtime kwargs: " + "; ".join([str(e) for e in runtime_kwargs_validation.get("errors", [])])
        )
    runtime_kwargs = dict(runtime_kwargs_validation.get("normalized", {}))

    if str(model_name) == "AutoHINT":
        backend_l = str(backend).strip().lower()
        if backend_l == "optuna":
            raise ValueError("AutoHINT does not support backend='optuna'. Use backend='ray'.")
        if not isinstance(model_params.get("config"), dict):
            model_params["config"] = {"reconciliation": "BottomUp"}
        else:
            model_params["config"] = dict(model_params["config"])
            model_params["config"].setdefault("reconciliation", "BottomUp")
        if "cls_model" not in model_params:
            try:
                from neuralforecast.models import NHITS

                model_params["cls_model"] = NHITS
            except Exception as e:
                raise ValueError(f"AutoHINT default cls_model resolution failed: {e}") from e
        if "S" not in model_params:
            n_series = max(1, int(df[settings.id_col].nunique()))
            model_params["S"] = [[1.0 if i == j else 0.0 for j in range(n_series)] for i in range(n_series)]
        if not isinstance(model_params.get("S"), (list, tuple)):
            raise ValueError("AutoHINT requires matrix-like S. e.g. [[1,0],[0,1]]")

    validate_payload = dict(model_params)
    validate_payload.update(
        {
            "backend": backend,
            "num_samples": int(num_samples),
            "seed": int(seed),
            "loss_name": str(loss_name),
            "valid_loss_name": (str(valid_loss_name) if valid_loss_name else None),
            "search_alg_name": (str(search_alg_name) if search_alg_name else None),
            "cpus": (_safe_int(cpus, 0) if cpus is not None else None),
            "gpus": (_safe_int(gpus, 0) if gpus is not None else None),
            "refit_with_val": bool(refit_with_val),
            "verbose": bool(verbose),
            "strict_exog": bool(strict_exog),
            "run_cross_validation": bool(run_cross_validation),
            "local_scaler_type": (str(local_scaler_type) if local_scaler_type else None),
            "local_static_scaler_type": (str(local_static_scaler_type) if local_static_scaler_type else None),
            "futr_exog_list": list(exog_input.get("futr_exog", [])),
            "hist_exog_list": list(exog_input.get("hist_exog", [])),
            "stat_exog_list": list(exog_input.get("stat_exog", [])),
            "nf_fit_kwargs": dict(runtime_kwargs.get("nf_fit_kwargs", {})),
            "nf_predict_kwargs": dict(runtime_kwargs.get("nf_predict_kwargs", {})),
            "nf_cross_validation_kwargs": dict(runtime_kwargs.get("nf_cross_validation_kwargs", {})),
            "nf_save_kwargs": dict(runtime_kwargs.get("nf_save_kwargs", {})),
            "nf_load_kwargs": dict(runtime_kwargs.get("nf_load_kwargs", {})),
            "nf_predict_insample_kwargs": dict(runtime_kwargs.get("nf_predict_insample_kwargs", {})),
        }
    )
    validate_payload = {k: v for k, v in validate_payload.items() if v is not None}
    try:
        from .registry import get_adapter

        v_report = get_adapter("neuralforecast_auto").validate(model_name=model_name, model_params=validate_payload)
        if not bool(v_report.get("ok", False)):
            errs = "; ".join([str(x) for x in v_report.get("errors", [])])
            raise ValueError(f"model param validation failed: {errs}")
        v_warnings = [str(x) for x in v_report.get("warnings", [])]
        if v_warnings:
            logger.warning("model param warnings: " + "; ".join(v_warnings))
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"model param validation skipped: {e}")

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    run_dir = settings.artifact_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fit_df, _, exog = prepare_nf_frames(df, exog=exog_input)

    model = build_automodel(
        model_name=model_name,
        h=h,
        exog=exog,
        backend=backend,
        num_samples=num_samples,
        seed=seed,
        loss_name=loss_name,
        valid_loss_name=str(loss_name),
        search_alg_name=str(search_alg_name) if search_alg_name else None,
        cpus=_safe_int(cpus, 0) if cpus is not None else None,
        gpus=_safe_int(gpus, 0) if gpus is not None else None,
        refit_with_val=refit_with_val,
        verbose=verbose,
        alias=str(alias) if alias is not None else None,
        callbacks=callbacks,
        model_kwargs=model_params,
    )

    NeuralForecast, _ = _load_neuralforecast_runtime()
    nf_kwargs: dict[str, Any] = {"models": [model], "freq": freq}
    if local_scaler_type:
        nf_kwargs["local_scaler_type"] = str(local_scaler_type)
        nf_kwargs["local_static_scaler_type"] = str(local_scaler_type)
    nf = NeuralForecast(**nf_kwargs)

    _configure_torch_runtime()

    logger.info(
        f"fit start model={model_name} run_id={run_id} h={h} backend={backend} num_samples={num_samples} "
        f"rows={len(fit_df)} cols={len(fit_df.columns)} exog={exog} local_scaler_type={local_scaler_type} "
        f"extra={model_params} "
        f"nf_fit_kwargs={runtime_kwargs.get('nf_fit_kwargs', {})}"
    )
    fit_kwargs_runtime = _resolve_nf_fit_kwargs_for_runtime(
        fit_df=fit_df,
        fit_kwargs_in=dict(runtime_kwargs.get("nf_fit_kwargs", {})),
    )
    fit_kwargs_log = dict(fit_kwargs_runtime)
    if isinstance(fit_kwargs_log.get("static_df"), pd.DataFrame):
        sdf = fit_kwargs_log["static_df"]
        fit_kwargs_log["static_df"] = {
            "rows": int(len(sdf)),
            "columns": [str(c) for c in sdf.columns],
        }
    logger.info(f"nf.fit resolved kwargs={fit_kwargs_log}")
    nf.fit(df=fit_df, **fit_kwargs_runtime)

    cv_info: dict[str, Any] = {}
    cv_kwargs = dict(runtime_kwargs.get("nf_cross_validation_kwargs", {}))
    cv_enabled = run_cross_validation or bool(cv_kwargs)
    if cv_enabled:
        cv_data_kwargs = cv_kwargs.pop("data_kwargs", {})
        if not isinstance(cv_data_kwargs, dict):
            cv_data_kwargs = {}
        cv_df = nf.cross_validation(df=fit_df, **cv_kwargs, **cv_data_kwargs)
        cv_path = run_dir / "cross_validation.parquet"
        cv_df.to_parquet(cv_path, index=False)
        cv_info = {
            "enabled": True,
            "path": str(cv_path),
            "rows": int(len(cv_df)),
            "columns": [str(c) for c in cv_df.columns],
        }
    else:
        cv_info = {"enabled": False}

    save_kwargs = {"overwrite": True, "save_dataset": False}
    save_kwargs.update(dict(runtime_kwargs.get("nf_save_kwargs", {})))
    nf.save(path=str(run_dir), **save_kwargs)
    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "h": h,
        "freq": freq,
        "exog": exog,
        "backend": backend,
        "num_samples": num_samples,
        "seed": seed,
        "loss_name": loss_name,
        "valid_loss_name": valid_loss_name,
        "local_scaler_type": local_scaler_type,
        "local_static_scaler_type": local_static_scaler_type,
        "search_alg_name": search_alg_name,
        "cpus": cpus,
        "gpus": gpus,
        "refit_with_val": refit_with_val,
        "verbose": verbose,
        "alias": alias,
        "callbacks": _json_list(callbacks_raw),
        "strict_exog": strict_exog,
        "model_exog_support": get_model_exog_support(model_name),
        "model_exog_support_table": model_exog_support_table(),
        "exog_warnings": exog_warnings,
        "explicit_exog_roles": sorted(list(explicit_exog_roles)),
        "nf_runtime_kwargs": runtime_kwargs,
        "nf_runtime_kwargs_raw": runtime_kwargs_raw,
        "cross_validation": cv_info,
        "model_params": model_params,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return TrainResult(run_id=run_id, model_name=model_name, artifact_path=run_dir, exog=exog)


def load_model(run_dir: Path, load_kwargs: dict[str, Any] | None = None):
    NeuralForecast, _ = _load_neuralforecast_runtime()
    kwargs = dict(load_kwargs or {})
    return NeuralForecast.load(path=str(run_dir), **kwargs)


def predict_with_model(
    nf: Any,
    df: pd.DataFrame,
    futr_df: pd.DataFrame | None = None,
    predict_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    _configure_torch_runtime()
    df = df.copy()
    kwargs = dict(predict_kwargs or {})
    kwargs.pop("df", None)
    kwargs.pop("futr_df", None)
    data_kwargs = kwargs.pop("data_kwargs", {})
    if not isinstance(data_kwargs, dict):
        data_kwargs = {}
    if futr_df is None:
        return nf.predict(df=df, **kwargs, **data_kwargs)
    return nf.predict(df=df, futr_df=futr_df.copy(), **kwargs, **data_kwargs)


def _collect_file_stats(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve()
    files: list[dict[str, Any]] = []
    ext_counts: dict[str, int] = {}
    total_bytes = 0
    if root.exists():
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            sz = int(p.stat().st_size)
            total_bytes += sz
            ext = p.suffix.lower() or "<no_ext>"
            ext_counts[ext] = int(ext_counts.get(ext, 0)) + 1
            files.append(
                {
                    "path": str(p.relative_to(root)),
                    "size_bytes": sz,
                }
            )
    return {
        "path": str(root),
        "exists": root.exists(),
        "file_count": int(len(files)),
        "total_bytes": int(total_bytes),
        "ext_counts": ext_counts,
        "files": files[:200],
    }


def save_load_analyze_model_bundle(
    run_id: str,
    source_dir: Path,
    save_path: str | None = None,
    run_save: bool = True,
    run_load: bool = True,
    run_analyze: bool = True,
    save_dataset: bool = False,
    save_overwrite: bool = True,
    load_check_predict: bool = False,
    insample_step_size: int = 1,
    save_kwargs: dict[str, Any] | None = None,
    load_kwargs: dict[str, Any] | None = None,
    predict_insample_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src = Path(source_dir).expanduser().resolve()
    store = Path(save_path).expanduser().resolve() if save_path else src
    out: dict[str, Any] = {
        "run_id": run_id,
        "source_dir": str(src),
        "store_path": str(store),
        "save": {},
        "load": {},
        "analyze": {},
    }

    if run_save:
        nf = load_model(src, load_kwargs=load_kwargs)
        store.mkdir(parents=True, exist_ok=True)
        resolved_save_kwargs = {"overwrite": bool(save_overwrite), "save_dataset": bool(save_dataset)}
        resolved_save_kwargs.update(dict(save_kwargs or {}))
        nf.save(path=str(store), **resolved_save_kwargs)
        stats = _collect_file_stats(store)
        out["save"] = {
            "ok": True,
            "path": str(store),
            "save_dataset": bool(resolved_save_kwargs.get("save_dataset", save_dataset)),
            "overwrite": bool(resolved_save_kwargs.get("overwrite", save_overwrite)),
            "file_count": int(stats.get("file_count", 0)),
            "total_bytes": int(stats.get("total_bytes", 0)),
        }

    if run_load:
        load_target = store if (run_save or save_path) else src
        nf_loaded = load_model(load_target, load_kwargs=load_kwargs)
        model_names = [type(m).__name__ for m in getattr(nf_loaded, "models", [])]
        load_info: dict[str, Any] = {
            "ok": True,
            "path": str(load_target),
            "model_count": int(len(model_names)),
            "model_names": model_names,
        }
        if load_check_predict:
            try:
                resolved_predict_insample_kwargs = {"step_size": max(1, int(insample_step_size))}
                resolved_predict_insample_kwargs.update(dict(predict_insample_kwargs or {}))
                insample = nf_loaded.predict_insample(**resolved_predict_insample_kwargs)
                load_info["predict_insample"] = {
                    "ok": True,
                    "rows": int(len(insample)),
                    "columns": [str(c) for c in insample.columns],
                    "kwargs": resolved_predict_insample_kwargs,
                }
            except Exception as e:
                load_info["predict_insample"] = {"ok": False, "error": str(e)}
        out["load"] = load_info

    if run_analyze:
        analyze_target = store if (run_save or save_path) else src
        out["analyze"] = _collect_file_stats(analyze_target)

    return out
