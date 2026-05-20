from __future__ import annotations

import argparse
import os
import dataclasses
import importlib
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import inspect
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from loto_forecast.infra.meta_store import mark_model_run_end, upsert_model_run, write_log_run_history, write_resource_samples
from loto_forecast.infra.monitoring import ResourceMonitor, generate_run_id
from .config import ResourcesConfig
from .context import start_run
from .db.postgres_copy import copy_dataframe_to_postgres
from .utils import safe_ident


@dataclass(frozen=True)
class ExogBuildSpec:
    profile: str = "local"
    env: str = "LOCAL"
    host: str = "127.0.0.1"
    port: int = 5432
    user: str = "loto"
    password: str = ""
    database: str = "loto"

    source_schema: str = "dataset"
    source_table: str = "loto_y_ts"
    source_where: str | None = None

    target_schema: str = "exog"
    target_table: str = "loto_y_ts_exog"
    if_exists: str = "replace"

    group_cols: tuple[str, ...] = ("loto", "unique_id", "ts_type")
    time_col: str = "ds"
    target_col: str = "y"

    parallel_workers: int = 4
    enable_gpu_compute: bool = True
    enable_anomaly_features: bool = False
    create_postgres_index: bool = True
    postgres_write_mode: str = "copy"
    postgres_copy_chunk_rows: int = 50000
    profile_stages: bool = False
    row_batch_size: int = 5000
    max_groups_per_batch: int = 64
    feature_families: tuple[str, ...] = ("base", "hist", "stat")
    pyod_codegen_yaml: str | None = "./docs/lib_docs/pyod_all_codegen.yaml"
    pyod_detectors: tuple[str, ...] = ("ECOD", "IForest", "COPOD")
    pyod_contamination: float = 0.1
    anomaly_min_train_size: int = 20
    anomaly_rolling_window: int = 14
    enable_merlion_features: bool = False
    merlion_codegen_yaml: str | None = (
        "./docs/lib_docs/merlion_dashboard_selected_codegen_details.yaml"
    )
    merlion_models: tuple[str, ...] = ("iforest", "lof", "spectral_residual", "stat_threshold")
    merlion_contamination: float = 0.1
    merlion_min_train_size: int = 30
    merlion_n_estimators: int = 100
    merlion_max_n_samples: int = 512
    merlion_random_state: int = 42
    enable_pypots_features: bool = False
    pypots_codegen_yaml: str | None = "./docs/lib_docs/pypots_all_codegen.yaml"
    pypots_models: tuple[str, ...] = ("transformer", "saits")
    pypots_anomaly_rate: float = 0.1
    pypots_window_size: int = 32
    pypots_min_train_windows: int = 20
    pypots_epochs: int = 2
    pypots_batch_size: int = 32
    enable_tsfel_features: bool = False
    tsfel_codegen_yaml: str | None = "./docs/lib_docs/tsfel_all_codegen.yaml"
    tsfel_domains: tuple[str, ...] = ("statistical", "temporal", "spectral")
    tsfel_max_features: int = 64
    tsfel_window_size: int = 32
    tsfel_min_train_windows: int = 20
    tsfel_fill_method: str = "ffill"
    tsfel_sampling_frequency: float = 1.0
    enable_autogluon_features: bool = False
    autogluon_codegen_yaml: str | None = "./docs/lib_docs/autogluon__internal__all_codegen.yaml"
    autogluon_generators: tuple[str, ...] = ("automl_pipeline",)
    autogluon_window_size: int = 32
    autogluon_min_train_windows: int = 20
    autogluon_fill_method: str = "ffill"
    autogluon_max_features: int = 64
    enable_stumpy_features: bool = False
    stumpy_codegen_yaml: str | None = "./docs/lib_docs/stumpy_all_codegen.yaml"
    stumpy_window_size: int = 32
    stumpy_min_train_windows: int = 20
    stumpy_fill_method: str = "ffill"
    stumpy_discord_quantile: float = 0.98
    enable_tsfresh_features: bool = False
    tsfresh_codegen_yaml: str | None = "./docs/lib_docs/tsfresh_all_codegen.yaml"
    tsfresh_feature_set: str = "minimal"
    tsfresh_window_size: int = 32
    tsfresh_min_train_windows: int = 20
    tsfresh_fill_method: str = "ffill"
    tsfresh_max_features: int = 64
    tsfresh_n_jobs: int = 0

    sampling_interval_sec: float = 1.0
    lib_docs_dir: str | None = "./docs/lib_docs"


def _make_engine(spec: ExogBuildSpec) -> Engine:
    url = f"postgresql+psycopg2://{spec.user}:{spec.password}@{spec.host}:{spec.port}/{spec.database}"
    return create_engine(url, pool_pre_ping=True)


def _summarize_lib_docs(lib_docs_dir: str | None) -> dict[str, Any]:
    if not lib_docs_dir:
        return {}
    p = Path(lib_docs_dir)
    if not p.exists() or not p.is_dir():
        return {"lib_docs_dir": str(p), "exists": False}

    rows: list[dict[str, Any]] = []
    for f in sorted(p.glob("*_all_codegen.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        library = (
            str(data.get("rows", [{}])[0].get("library", f.stem.replace("_all_codegen", "")))
            if isinstance(data.get("rows"), list) and data.get("rows")
            else f.stem.replace("_all_codegen", "")
        )
        rows.append(
            {
                "file": f.name,
                "library": library,
                "count": int(data.get("count", 0) or 0),
            }
        )
    return {
        "lib_docs_dir": str(p),
        "exists": True,
        "file_count": len(rows),
        "libraries": sorted(list({r["library"] for r in rows})),
        "total_codegen_rows": int(sum(r["count"] for r in rows)),
    }


def _normalize_detector_name(name: str) -> str:
    key = "".join(ch for ch in str(name or "").strip() if ch.isalnum() or ch == "_")
    aliases = {
        "ecod": "ECOD",
        "iforest": "IForest",
        "copod": "COPOD",
        "hbos": "HBOS",
        "knn": "KNN",
        "lof": "LOF",
        "ocsvm": "OCSVM",
        "pca": "PCA",
    }
    return aliases.get(key.lower(), key)


def _normalize_detector_names(detectors: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in detectors:
        name = _normalize_detector_name(raw)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _summarize_pyod_codegen(path: str | None, requested_detectors: tuple[str, ...]) -> dict[str, Any]:
    normalized = _normalize_detector_names(requested_detectors)
    if not path:
        return {"path": None, "exists": False, "requested_detectors": list(normalized)}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested_detectors": list(normalized)}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested_detectors": list(normalized),
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    detector_classes: list[str] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("type", "")).lower() != "class":
                continue
            full_path = str(r.get("path", ""))
            if not full_path.startswith("pyod.models."):
                continue
            name = _normalize_detector_name(str(r.get("name", "")))
            if name:
                detector_classes.append(name)

    available = sorted(set(detector_classes))
    available_set = set(available)
    requested_available = [d for d in normalized if d in available_set]
    requested_missing = [d for d in normalized if d not in available_set]

    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "detector_class_count": len(available),
        "requested_detectors": list(normalized),
        "requested_available": requested_available,
        "requested_missing_in_codegen": requested_missing,
    }


def _normalize_merlion_model_name(name: str) -> str:
    key = "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum() or ch == "_")
    aliases = {
        "iforest": "iforest",
        "isolationforest": "iforest",
        "isolation_forest": "iforest",
        "lof": "lof",
        "spectralresidual": "spectral_residual",
        "spectral_residual": "spectral_residual",
        "sr": "spectral_residual",
        "statthreshold": "stat_threshold",
        "stat_threshold": "stat_threshold",
        "threshold": "stat_threshold",
    }
    return aliases.get(key, key)


def _normalize_merlion_model_names(models: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in models:
        name = _normalize_merlion_model_name(raw)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _summarize_merlion_codegen(path: str | None, requested_models: tuple[str, ...]) -> dict[str, Any]:
    normalized = _normalize_merlion_model_names(requested_models)
    if not path:
        return {"path": None, "exists": False, "requested_models": list(normalized)}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested_models": list(normalized)}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested_models": list(normalized),
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    available: list[str] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            model_name = _normalize_merlion_model_name(str(r.get("name", "")))
            if model_name:
                available.append(model_name)
    available_set = set(available)

    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "requested_models": list(normalized),
        "requested_available": [m for m in normalized if m in available_set],
        "requested_missing_in_codegen": [m for m in normalized if m not in available_set],
    }


def _normalize_pypots_model_name(name: str) -> str:
    key = "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum() or ch == "_")
    aliases = {
        "transformer": "transformer",
        "saits": "saits",
        "dlinear": "dlinear",
        "timesnet": "timesnet",
    }
    return aliases.get(key, key)


def _normalize_pypots_model_names(models: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in models:
        name = _normalize_pypots_model_name(raw)
        if name and name not in out:
            out.append(name)
    return tuple(out)


def _summarize_pypots_codegen(path: str | None, requested_models: tuple[str, ...]) -> dict[str, Any]:
    normalized = _normalize_pypots_model_names(requested_models)
    if not path:
        return {"path": None, "exists": False, "requested_models": list(normalized)}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested_models": list(normalized)}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested_models": list(normalized),
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    available: list[str] = []
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            if str(r.get("type", "")).lower() != "class":
                continue
            full_path = str(r.get("path", ""))
            if not full_path.startswith("pypots.anomaly_detection."):
                continue
            m = full_path.split(".")
            if len(m) >= 3:
                available.append(_normalize_pypots_model_name(m[2]))

    available_set = set([x for x in available if x])
    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "requested_models": list(normalized),
        "requested_available": [m for m in normalized if m in available_set],
        "requested_missing_in_codegen": [m for m in normalized if m not in available_set],
    }


def _normalize_tsfel_domain_name(name: str) -> str:
    key = "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum() or ch == "_")
    aliases = {
        "statistical": "statistical",
        "stats": "statistical",
        "temporal": "temporal",
        "time": "temporal",
        "spectral": "spectral",
        "frequency": "spectral",
        "fractal": "fractal",
    }
    return aliases.get(key, key)


def _normalize_tsfel_domain_names(domains: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in domains:
        d = _normalize_tsfel_domain_name(raw)
        if d and d not in out:
            out.append(d)
    return tuple(out)


def _summarize_tsfel_codegen(path: str | None, requested_domains: tuple[str, ...], max_features: int) -> dict[str, Any]:
    normalized = _normalize_tsfel_domain_names(requested_domains)
    if not path:
        return {"path": None, "exists": False, "requested_domains": list(normalized)}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested_domains": list(normalized)}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested_domains": list(normalized),
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    domain_counts: dict[str, int] = {}
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            top_group = str(r.get("top_group", ""))
            module = str(r.get("module", ""))
            if top_group != "tsfel.feature_extraction" and not module.startswith("tsfel.feature_extraction"):
                continue
            # Infer likely domain from function name when possible.
            name = str(r.get("name", "")).lower()
            inferred = None
            for d in normalized:
                if d in name:
                    inferred = d
                    break
            if inferred is not None:
                domain_counts[inferred] = domain_counts.get(inferred, 0) + 1

    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "requested_domains": list(normalized),
        "max_features": int(max_features),
        "domain_counts_hint": domain_counts,
    }


def _normalize_autogluon_generator_name(name: str) -> str:
    key = "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum() or ch == "_")
    aliases = {
        "automl_pipeline": "automl_pipeline",
        "automlpipeline": "automl_pipeline",
        "pipeline": "automl_pipeline",
    }
    return aliases.get(key, key)


def _normalize_autogluon_generator_names(generators: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in generators:
        g = _normalize_autogluon_generator_name(raw)
        if g and g not in out:
            out.append(g)
    return tuple(out)


def _summarize_autogluon_codegen(path: str | None, requested_generators: tuple[str, ...]) -> dict[str, Any]:
    normalized = _normalize_autogluon_generator_names(requested_generators)
    if not path:
        return {"path": None, "exists": False, "requested_generators": list(normalized)}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested_generators": list(normalized)}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested_generators": list(normalized),
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    feature_related = 0
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            top_group = str(r.get("top_group", ""))
            module = str(r.get("module", ""))
            if (
                top_group.startswith("autogluon.features")
                or top_group == "autogluon.common"
                or module.startswith("autogluon.features.")
                or module.startswith("autogluon.common.features.")
            ):
                feature_related += 1

    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "requested_generators": list(normalized),
        "feature_related_rows_hint": int(feature_related),
    }


def _summarize_stumpy_codegen(path: str | None, window_size: int, discord_quantile: float) -> dict[str, Any]:
    requested = {
        "window_size": int(window_size),
        "discord_quantile": float(discord_quantile),
    }
    if not path:
        return {"path": None, "exists": False, "requested": requested}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested": requested}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested": requested,
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    stumpy_related = 0
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            top_group = str(r.get("top_group", ""))
            module = str(r.get("module", ""))
            full_path = str(r.get("path", ""))
            if top_group.startswith("stumpy") or module.startswith("stumpy.") or full_path.startswith("stumpy."):
                stumpy_related += 1

    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "requested": requested,
        "stumpy_related_rows_hint": int(stumpy_related),
    }


def _normalize_tsfresh_feature_set(name: str) -> str:
    key = "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum() or ch == "_")
    aliases = {
        "minimal": "minimal",
        "minimalfcparameters": "minimal",
        "efficient": "efficient",
        "efficientfcparameters": "efficient",
        "comprehensive": "comprehensive",
        "comprehensivefcparameters": "comprehensive",
        "index": "index_based",
        "indexbased": "index_based",
        "index_based": "index_based",
        "time": "time_based",
        "timebased": "time_based",
        "time_based": "time_based",
    }
    return aliases.get(key, key)


def _summarize_tsfresh_codegen(path: str | None, feature_set: str, max_features: int, n_jobs: int) -> dict[str, Any]:
    requested = {
        "feature_set": _normalize_tsfresh_feature_set(feature_set),
        "max_features": int(max_features),
        "n_jobs": int(n_jobs),
    }
    if not path:
        return {"path": None, "exists": False, "requested": requested}

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "requested": requested}

    try:
        payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {
            "path": str(p),
            "exists": True,
            "requested": requested,
            "error": str(e),
        }

    rows = payload.get("rows") if isinstance(payload, dict) else None
    extraction_related = 0
    calculator_rows = 0
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            top_group = str(r.get("top_group", ""))
            module = str(r.get("module", ""))
            full_path = str(r.get("path", ""))
            if (
                top_group.startswith("tsfresh.feature_extraction")
                or module.startswith("tsfresh.feature_extraction.")
                or full_path.startswith("tsfresh.feature_extraction.")
            ):
                extraction_related += 1
            if module == "tsfresh.feature_extraction.feature_calculators":
                calculator_rows += 1

    return {
        "path": str(p),
        "exists": True,
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "row_count": int(payload.get("count", 0) or 0) if isinstance(payload, dict) else 0,
        "requested": requested,
        "extraction_related_rows_hint": int(extraction_related),
        "feature_calculator_rows_hint": int(calculator_rows),
    }


def _safe_table_ref(schema: str, table: str) -> str:
    return f'"{safe_ident(schema)}"."{safe_ident(table)}"'


def _read_source(spec: ExogBuildSpec, engine: Engine) -> pd.DataFrame:
    table_ref = _safe_table_ref(spec.source_schema, spec.source_table)
    cols = set(spec.group_cols) | {spec.time_col, spec.target_col}
    optional_cols = {"proc_seconds", "exec_ts", f"{spec.source_table}_row_id", "loto_y_ts_row_id", "row_id"}
    available_cols = {
        str(col["name"])
        for col in sqlalchemy_inspect(engine).get_columns(safe_ident(spec.source_table), schema=safe_ident(spec.source_schema))
    }
    selected_cols = [c for c in [*spec.group_cols, spec.time_col, spec.target_col] if c in available_cols]
    selected_cols.extend(sorted(c for c in optional_cols if c in available_cols and c not in selected_cols))
    missing = cols - set(selected_cols)
    if missing:
        raise ValueError(f"missing source columns in table metadata: {sorted(missing)}")
    select_list = ", ".join(f'"{safe_ident(c)}"' for c in selected_cols)
    sql = f"SELECT {select_list} FROM {table_ref}"
    if spec.source_where:
        sql += f" WHERE {spec.source_where}"
    df = pd.read_sql(sql, engine)
    return df


def _validate_input(df: pd.DataFrame, spec: ExogBuildSpec) -> None:
    required = set(spec.group_cols) | {spec.time_col, spec.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in source: {sorted(missing)}")
    if not (0.0 < float(spec.pyod_contamination) < 0.5):
        raise ValueError("pyod_contamination must be in (0.0, 0.5)")
    if int(spec.anomaly_min_train_size) < 3:
        raise ValueError("anomaly_min_train_size must be >= 3")
    if int(spec.anomaly_rolling_window) < 3:
        raise ValueError("anomaly_rolling_window must be >= 3")
    if not (0.0 < float(spec.merlion_contamination) < 0.5):
        raise ValueError("merlion_contamination must be in (0.0, 0.5)")
    if int(spec.merlion_min_train_size) < 3:
        raise ValueError("merlion_min_train_size must be >= 3")
    if int(spec.merlion_n_estimators) < 10:
        raise ValueError("merlion_n_estimators must be >= 10")
    if int(spec.merlion_max_n_samples) < 32:
        raise ValueError("merlion_max_n_samples must be >= 32")
    if not (0.0 < float(spec.pypots_anomaly_rate) < 0.5):
        raise ValueError("pypots_anomaly_rate must be in (0.0, 0.5)")
    if int(spec.pypots_window_size) < 8:
        raise ValueError("pypots_window_size must be >= 8")
    if int(spec.pypots_min_train_windows) < 5:
        raise ValueError("pypots_min_train_windows must be >= 5")
    if int(spec.pypots_epochs) < 1:
        raise ValueError("pypots_epochs must be >= 1")
    if int(spec.pypots_batch_size) < 4:
        raise ValueError("pypots_batch_size must be >= 4")
    if int(spec.tsfel_max_features) < 1:
        raise ValueError("tsfel_max_features must be >= 1")
    if int(spec.tsfel_window_size) < 8:
        raise ValueError("tsfel_window_size must be >= 8")
    if int(spec.tsfel_min_train_windows) < 5:
        raise ValueError("tsfel_min_train_windows must be >= 5")
    if float(spec.tsfel_sampling_frequency) <= 0:
        raise ValueError("tsfel_sampling_frequency must be > 0")
    if int(spec.autogluon_window_size) < 8:
        raise ValueError("autogluon_window_size must be >= 8")
    if int(spec.autogluon_min_train_windows) < 5:
        raise ValueError("autogluon_min_train_windows must be >= 5")
    if int(spec.autogluon_max_features) < 1:
        raise ValueError("autogluon_max_features must be >= 1")
    if str(spec.autogluon_fill_method).strip().lower() not in {"ffill", "bfill", "interpolate", "zero", "mean"}:
        raise ValueError("autogluon_fill_method must be one of ffill,bfill,interpolate,zero,mean")
    if int(spec.stumpy_window_size) < 8:
        raise ValueError("stumpy_window_size must be >= 8")
    if int(spec.stumpy_min_train_windows) < 5:
        raise ValueError("stumpy_min_train_windows must be >= 5")
    if str(spec.stumpy_fill_method).strip().lower() not in {"ffill", "bfill", "interpolate", "zero", "mean"}:
        raise ValueError("stumpy_fill_method must be one of ffill,bfill,interpolate,zero,mean")
    if not (0.5 <= float(spec.stumpy_discord_quantile) < 1.0):
        raise ValueError("stumpy_discord_quantile must be in [0.5, 1.0)")
    if _normalize_tsfresh_feature_set(spec.tsfresh_feature_set) not in {
        "minimal",
        "efficient",
        "comprehensive",
        "index_based",
        "time_based",
    }:
        raise ValueError("tsfresh_feature_set must be one of minimal,efficient,comprehensive,index_based,time_based")
    if int(spec.tsfresh_window_size) < 8:
        raise ValueError("tsfresh_window_size must be >= 8")
    if int(spec.tsfresh_min_train_windows) < 5:
        raise ValueError("tsfresh_min_train_windows must be >= 5")
    if str(spec.tsfresh_fill_method).strip().lower() not in {"ffill", "bfill", "interpolate", "zero", "mean"}:
        raise ValueError("tsfresh_fill_method must be one of ffill,bfill,interpolate,zero,mean")
    if int(spec.tsfresh_max_features) < 1:
        raise ValueError("tsfresh_max_features must be >= 1")
    if int(spec.tsfresh_n_jobs) < 0:
        raise ValueError("tsfresh_n_jobs must be >= 0")


def _base_feature_frame(df: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    out = df.copy()
    out[spec.time_col] = pd.to_datetime(out[spec.time_col])
    out = out.sort_values(list(spec.group_cols) + [spec.time_col]).reset_index(drop=True)

    ds = pd.to_datetime(out[spec.time_col])
    out["feat_year"] = ds.dt.year
    out["feat_month"] = ds.dt.month
    out["feat_day"] = ds.dt.day
    out["feat_dayofweek"] = ds.dt.dayofweek
    out["feat_weekofyear"] = ds.dt.isocalendar().week.astype(int)
    out["feat_dayofyear"] = ds.dt.dayofyear
    out["feat_is_weekend"] = (out["feat_dayofweek"] >= 5).astype(int)
    out["feat_is_month_start"] = ds.dt.is_month_start.astype(int)
    out["feat_is_month_end"] = ds.dt.is_month_end.astype(int)

    out["feat_dow_sin"] = np.sin(2 * np.pi * out["feat_dayofweek"] / 7.0)
    out["feat_dow_cos"] = np.cos(2 * np.pi * out["feat_dayofweek"] / 7.0)
    out["feat_month_sin"] = np.sin(2 * np.pi * out["feat_month"] / 12.0)
    out["feat_month_cos"] = np.cos(2 * np.pi * out["feat_month"] / 12.0)

    g = out.groupby(list(spec.group_cols), sort=False)
    out["feat_days_since_first"] = (out[spec.time_col] - g[spec.time_col].transform("min")).dt.days.astype(int)
    out["feat_row_no_in_group"] = g.cumcount() + 1

    if "proc_seconds" in out.columns:
        out["feat_proc_seconds"] = pd.to_numeric(out["proc_seconds"], errors="coerce")
    if "exec_ts" in out.columns:
        exec_ts = pd.to_datetime(out["exec_ts"], errors="coerce")
        out["feat_exec_lag_sec"] = (exec_ts - out[spec.time_col]).dt.total_seconds()

    return out


def _trend_slope(y: pd.Series) -> float:
    z = pd.to_numeric(y, errors="coerce").astype(float)
    mask = z.notna().values
    if mask.sum() < 2:
        return np.nan
    arr = z.values[mask]
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _static_feature_frame(df: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    g = df.groupby(keys, sort=False)[spec.target_col]
    agg = g.agg(["count", "mean", "std", "min", "max", "median"]).reset_index()
    agg = agg.rename(
        columns={
            "count": "stat_y_count",
            "mean": "stat_y_mean",
            "std": "stat_y_std",
            "min": "stat_y_min",
            "max": "stat_y_max",
            "median": "stat_y_median",
        }
    )

    q = g.quantile([0.25, 0.75]).unstack().reset_index().rename(columns={0.25: "stat_y_q25", 0.75: "stat_y_q75"})
    slope = g.apply(_trend_slope).reset_index().rename(columns={spec.target_col: "stat_y_trend_slope"})

    out = agg.merge(q, on=keys, how="left").merge(slope, on=keys, how="left")
    out["stat_y_iqr"] = out["stat_y_q75"] - out["stat_y_q25"]
    out["stat_y_cv"] = out["stat_y_std"] / (out["stat_y_mean"].abs() + 1e-9)
    return out


def _gpu_features_for_values(values: np.ndarray, enabled: bool) -> dict[str, np.ndarray]:
    if not enabled:
        return {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        arr = np.nan_to_num(values.astype(np.float32), nan=0.0)
        t = torch.tensor(arr, dtype=torch.float32, device="cuda")
        idx = torch.arange(1, t.shape[0] + 1, dtype=torch.float32, device="cuda")
        cummean = (torch.cumsum(t, dim=0) / idx).detach().cpu().numpy()
        z = ((t - torch.mean(t)) / torch.clamp(torch.std(t, unbiased=False), min=1e-6)).detach().cpu().numpy()
        return {
            "feat_gpu_cummean_y": cummean,
            "feat_gpu_zscore_y": z,
        }
    except Exception:
        return {}


def _rolling_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def _load_pyod_detector_classes(detectors: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _normalize_detector_names(detectors):
        try:
            module = importlib.import_module(f"pyod.models.{name.lower()}")
            cls = getattr(module, name, None)
            if cls is not None:
                out[name] = cls
        except Exception:
            continue
    return out


def _fit_pyod_scores_flags(
    detector_class: Any,
    values_2d: np.ndarray,
    contamination: float,
) -> tuple[np.ndarray, np.ndarray]:
    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(detector_class)
        if "contamination" in sig.parameters:
            kwargs["contamination"] = float(contamination)
        if "random_state" in sig.parameters:
            kwargs["random_state"] = 42
    except Exception:
        pass

    detector = detector_class(**kwargs)
    detector.fit(values_2d)

    if hasattr(detector, "decision_scores_"):
        scores = np.asarray(detector.decision_scores_, dtype=float)
    else:
        scores = np.asarray(detector.decision_function(values_2d), dtype=float)

    labels = getattr(detector, "labels_", None)
    if labels is None:
        thr = np.nanquantile(scores, 1.0 - float(contamination))
        labels = (scores >= thr).astype(float)
    else:
        labels = np.asarray(labels, dtype=float)
    return scores, labels


def _timeseries_to_series(ts_obj: Any) -> pd.Series:
    if ts_obj is None:
        return pd.Series(dtype=float)
    if not hasattr(ts_obj, "to_pd"):
        return pd.Series(dtype=float)
    obj = ts_obj.to_pd()
    if isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce")
    elif isinstance(obj, pd.DataFrame):
        if obj.empty:
            return pd.Series(dtype=float)
        s = pd.to_numeric(obj.iloc[:, 0], errors="coerce")
    else:
        return pd.Series(dtype=float)
    return s.astype(float)


def _load_merlion_model(model_key: str, spec: ExogBuildSpec) -> Any:
    model_key = _normalize_merlion_model_name(model_key)
    if model_key == "iforest":
        from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig

        cfg = IsolationForestConfig(
            max_n_samples=int(spec.merlion_max_n_samples),
            n_estimators=int(spec.merlion_n_estimators),
            n_jobs=1,
            random_state=int(spec.merlion_random_state),
        )
        return IsolationForest(cfg)
    if model_key == "lof":
        from merlion.models.anomaly.lof import LOF, LOFConfig

        cfg = LOFConfig(
            n_neighbors=min(20, max(5, int(spec.merlion_min_train_size // 2))),
            contamination=float(spec.merlion_contamination),
            n_jobs=1,
        )
        return LOF(cfg)
    if model_key == "spectral_residual":
        from merlion.models.anomaly.spectral_residual import SpectralResidual, SpectralResidualConfig

        cfg = SpectralResidualConfig()
        return SpectralResidual(cfg)
    if model_key == "stat_threshold":
        from merlion.models.anomaly.stat_threshold import StatThreshold, StatThresholdConfig

        cfg = StatThresholdConfig()
        return StatThreshold(cfg)
    raise ValueError(f"unsupported merlion model: {model_key}")


def _fit_merlion_score_series(
    model_key: str,
    values: np.ndarray,
    spec: ExogBuildSpec,
) -> np.ndarray:
    from merlion.utils import TimeSeries

    arr = np.asarray(values, dtype=np.float32)
    idx = pd.date_range("2000-01-01", periods=arr.shape[0], freq="D")
    ts = TimeSeries.from_pd(pd.Series(arr, index=idx))
    model = _load_merlion_model(model_key, spec)
    model.train(ts)
    score_ts = model.get_anomaly_score(ts)
    score_s = _timeseries_to_series(score_ts)
    score_values = score_s.to_numpy(dtype=float)

    if score_values.shape[0] == arr.shape[0]:
        return score_values
    if score_values.shape[0] == 0:
        return np.full(arr.shape[0], np.nan, dtype=float)

    # Some models return shorter warm-up output; align to the tail.
    out = np.full(arr.shape[0], np.nan, dtype=float)
    out[-score_values.shape[0] :] = score_values
    return out


def _merlion_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    models = _normalize_merlion_model_names(spec.merlion_models)

    for model_key in models:
        out[f"hist_merlion_{model_key}_score"] = np.nan
        out[f"hist_merlion_{model_key}_flag"] = np.nan

    valid = y.notna()
    if valid.sum() < int(spec.merlion_min_train_size):
        return out

    try:
        import merlion  # noqa: F401
    except Exception:
        return out

    valid_pos = np.where(valid.to_numpy())[0]
    x_valid = y.iloc[valid_pos].to_numpy(dtype=np.float32)
    contamination = float(spec.merlion_contamination)

    for model_key in models:
        try:
            scores_valid = _fit_merlion_score_series(model_key, x_valid, spec)
        except Exception:
            continue

        if scores_valid.shape[0] != x_valid.shape[0]:
            continue

        score_s = pd.Series(np.nan, index=y.index, dtype=float)
        score_s.iloc[valid_pos] = scores_valid
        finite_scores = scores_valid[np.isfinite(scores_valid)]
        if finite_scores.size == 0:
            flag_s = pd.Series(np.nan, index=y.index, dtype=float)
        else:
            thr = float(np.nanquantile(finite_scores, 1.0 - contamination))
            flags_valid = (scores_valid >= thr).astype(float)
            flag_s = pd.Series(np.nan, index=y.index, dtype=float)
            flag_s.iloc[valid_pos] = flags_valid

        out[f"hist_merlion_{model_key}_score"] = score_s.shift(1).to_numpy()
        out[f"hist_merlion_{model_key}_flag"] = flag_s.shift(1).to_numpy()
    return out


def _resolve_pypots_device(enable_gpu_compute: bool) -> str:
    if not enable_gpu_compute:
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _build_sliding_windows_1d(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < window_size:
        return np.empty((0, window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)
    row_idx = np.arange(window_size - 1, arr.size, dtype=np.int64)
    return windows.astype(np.float32, copy=False), row_idx


def _load_pypots_model(model_key: str, spec: ExogBuildSpec, n_steps: int, n_features: int) -> Any:
    model_key = _normalize_pypots_model_name(model_key)
    device = _resolve_pypots_device(spec.enable_gpu_compute)
    common = {
        "n_steps": int(n_steps),
        "n_features": int(n_features),
        "anomaly_rate": float(spec.pypots_anomaly_rate),
        "batch_size": int(spec.pypots_batch_size),
        "epochs": int(spec.pypots_epochs),
        "patience": 1,
        "num_workers": 0,
        "device": device,
        "verbose": False,
    }

    if model_key == "transformer":
        from pypots.anomaly_detection.transformer import Transformer

        return Transformer(
            n_layers=1,
            d_model=16,
            n_heads=1,
            d_k=16,
            d_v=16,
            d_ffn=32,
            **common,
        )

    if model_key == "saits":
        from pypots.anomaly_detection.saits import SAITS

        return SAITS(
            n_layers=1,
            d_model=16,
            n_heads=1,
            d_k=16,
            d_v=16,
            d_ffn=32,
            **common,
        )

    if model_key == "dlinear":
        from pypots.anomaly_detection.dlinear import DLinear

        return DLinear(
            moving_avg_window_size=max(2, min(7, int(n_steps // 4))),
            d_model=16,
            **common,
        )

    if model_key == "timesnet":
        from pypots.anomaly_detection.timesnet import TimesNet

        return TimesNet(
            n_layers=1,
            top_k=2,
            d_model=16,
            d_ffn=32,
            n_kernels=2,
            **common,
        )

    raise ValueError(f"unsupported pypots model: {model_key}")


def _fit_pypots_window_flags(model_key: str, windows_3d: np.ndarray, spec: ExogBuildSpec) -> np.ndarray:
    model = _load_pypots_model(
        model_key=model_key, spec=spec, n_steps=windows_3d.shape[1], n_features=windows_3d.shape[2]
    )
    train_set = {"X": windows_3d}
    model.fit(train_set)
    pred = model.predict({"X": windows_3d})
    if not isinstance(pred, dict):
        raise ValueError("pypots predict output must be dict")
    raw = pred.get("anomaly_detection")
    if raw is None:
        raise ValueError("pypots predict output missing anomaly_detection")
    arr = np.asarray(raw, dtype=float)
    n_windows = windows_3d.shape[0]
    n_steps = windows_3d.shape[1]

    if arr.size == n_windows:
        flags = arr
    elif arr.size == n_windows * n_steps:
        flags = arr.reshape(n_windows, n_steps)[:, -1]
    elif arr.size > n_windows:
        flags = arr[-n_windows:]
    else:
        raise ValueError(f"unexpected pypots anomaly_detection shape: {arr.shape}")
    return (np.nan_to_num(flags, nan=0.0) > 0).astype(float)


def _pypots_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    shifted = y.shift(1)
    models = _normalize_pypots_model_names(spec.pypots_models)
    window = int(spec.pypots_window_size)

    out["hist_pypots_missing_ratio"] = shifted.isna().rolling(window, min_periods=1).mean().to_numpy()
    out["hist_pypots_missing_flag"] = shifted.isna().astype(float).to_numpy()

    for model_key in models:
        out[f"hist_pypots_{model_key}_score"] = np.nan
        out[f"hist_pypots_{model_key}_flag"] = np.nan

    windows_2d, row_idx = _build_sliding_windows_1d(shifted.to_numpy(dtype=np.float32), window_size=window)
    if windows_2d.shape[0] == 0:
        return out

    min_obs = max(4, window // 4)
    obs_counts = np.isfinite(windows_2d).sum(axis=1)
    keep = obs_counts >= min_obs
    if int(keep.sum()) < int(spec.pypots_min_train_windows):
        return out

    windows_fit = windows_2d[keep][:, :, np.newaxis]
    row_idx_fit = row_idx[keep]

    try:
        import pypots  # noqa: F401
    except Exception:
        return out

    for model_key in models:
        try:
            flags = _fit_pypots_window_flags(model_key=model_key, windows_3d=windows_fit, spec=spec)
        except Exception:
            continue

        flag_s = pd.Series(np.nan, index=y.index, dtype=float)
        flag_s.iloc[row_idx_fit] = flags
        score_s = flag_s.rolling(max(3, window // 4), min_periods=2).mean()
        out[f"hist_pypots_{model_key}_flag"] = flag_s.to_numpy()
        out[f"hist_pypots_{model_key}_score"] = score_s.to_numpy()

    return out


def _sanitize_feature_name(name: str) -> str:
    s = str(name or "").strip().lower()
    s = s.replace("%", "pct")
    out = []
    prev_us = False
    for ch in s:
        keep = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if keep:
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    val = "".join(out).strip("_")
    return val or "feature"


def _build_tsfel_config_and_feature_names(spec: ExogBuildSpec) -> tuple[dict[str, Any], list[str]]:
    import copy
    import tsfel

    cfg_raw = tsfel.get_features_by_domain()
    cfg = copy.deepcopy(cfg_raw)
    domains = _normalize_tsfel_domain_names(spec.tsfel_domains)
    requested = [d for d in domains if d in cfg]
    selected: list[tuple[str, str]] = []

    for _domain, feats in cfg.items():
        for f_name in feats:
            feats[f_name]["use"] = "no"

    blocked_tokens = {"ecdf", "lpcc", "mfcc", "spectrogram mean coefficient"}
    for domain in requested:
        feats = cfg.get(domain, {})
        for f_name in feats:
            low = str(f_name).strip().lower()
            if any(tok in low for tok in blocked_tokens):
                continue
            selected.append((domain, f_name))

    selected = selected[: max(1, int(spec.tsfel_max_features))]
    for domain, f_name in selected:
        cfg[domain][f_name]["use"] = "yes"

    feature_names = [f_name for _, f_name in selected]
    return cfg, feature_names


def _fill_series_for_tsfel(s: pd.Series, method: str) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    m = str(method or "ffill").strip().lower()
    if m == "zero":
        return s_num.fillna(0.0)
    if m == "mean":
        return s_num.fillna(float(s_num.mean())) if s_num.notna().any() else s_num.fillna(0.0)
    if m == "interpolate":
        return s_num.interpolate(limit_direction="both").fillna(0.0)
    if m == "bfill":
        return s_num.bfill().ffill().fillna(0.0)
    # default: ffill
    return s_num.ffill().bfill().fillna(0.0)


def _coerce_object_to_float(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return np.nan
        return float(np.nanmean(arr))
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return np.nan
        try:
            return float(text)
        except Exception:
            try:
                import ast

                parsed = ast.literal_eval(text)
                return _coerce_object_to_float(parsed)
            except Exception:
                return np.nan
    return np.nan


def _build_autogluon_window_summary(window_values: np.ndarray, missing_ratio: float) -> dict[str, float]:
    arr = np.asarray(window_values, dtype=float)
    n = arr.size
    idx = np.arange(n, dtype=float)
    slope = 0.0
    if n >= 2:
        try:
            slope = float(np.polyfit(idx, arr, 1)[0])
        except Exception:
            slope = 0.0
    q25 = float(np.nanquantile(arr, 0.25))
    q75 = float(np.nanquantile(arr, 0.75))
    return {
        "w_last": float(arr[-1]),
        "w_mean": float(np.nanmean(arr)),
        "w_std": float(np.nanstd(arr)),
        "w_min": float(np.nanmin(arr)),
        "w_max": float(np.nanmax(arr)),
        "w_median": float(np.nanmedian(arr)),
        "w_iqr": float(q75 - q25),
        "w_abs_energy": float(np.nansum(arr * arr)),
        "w_slope": float(slope),
        "w_missing_ratio": float(missing_ratio),
    }


def _autogluon_raw_feature_columns() -> list[str]:
    return [
        "hist_autogluon_raw_w_last",
        "hist_autogluon_raw_w_mean",
        "hist_autogluon_raw_w_std",
        "hist_autogluon_raw_w_min",
        "hist_autogluon_raw_w_max",
        "hist_autogluon_raw_w_median",
        "hist_autogluon_raw_w_iqr",
        "hist_autogluon_raw_w_abs_energy",
        "hist_autogluon_raw_w_slope",
        "hist_autogluon_raw_w_missing_ratio",
    ]


def _autogluon_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    shifted = y.shift(1)
    window = int(spec.autogluon_window_size)

    out["hist_autogluon_missing_ratio"] = shifted.isna().rolling(window, min_periods=1).mean().to_numpy()
    out["hist_autogluon_missing_flag"] = shifted.isna().astype(float).to_numpy()
    for c in _autogluon_raw_feature_columns():
        out[c] = np.nan

    windows_raw, row_idx = _build_sliding_windows_1d(shifted.to_numpy(dtype=np.float32), window_size=window)
    if windows_raw.shape[0] == 0:
        return out

    min_obs = max(4, window // 4)
    obs_counts = np.isfinite(windows_raw).sum(axis=1)
    keep = obs_counts >= min_obs
    if int(keep.sum()) < int(spec.autogluon_min_train_windows):
        return out

    filled = _fill_series_for_tsfel(shifted, method=spec.autogluon_fill_method)
    windows_filled, row_idx_filled = _build_sliding_windows_1d(filled.to_numpy(dtype=np.float32), window_size=window)
    if windows_filled.shape[0] == 0:
        return out

    # Align to filtered valid windows.
    valid_mask = keep[: min(len(keep), len(windows_filled))]
    windows_fit = windows_filled[: len(valid_mask)][valid_mask]
    row_idx_fit = row_idx_filled[: len(valid_mask)][valid_mask]
    miss_ratio = (1.0 - (obs_counts[: len(valid_mask)][valid_mask] / float(window))).astype(float)
    if windows_fit.shape[0] < int(spec.autogluon_min_train_windows):
        return out

    rows = [
        _build_autogluon_window_summary(win, missing_ratio=mr) for win, mr in zip(windows_fit, miss_ratio, strict=False)
    ]
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return out

    # Always expose deterministic raw window summary features.
    for raw_col in summary_df.columns:
        target_col = f"hist_autogluon_raw_{_sanitize_feature_name(raw_col)}"
        if target_col not in out.columns:
            out[target_col] = np.nan
        s = pd.Series(np.nan, index=out.index, dtype=float)
        s.iloc[row_idx_fit] = pd.to_numeric(summary_df[raw_col], errors="coerce").to_numpy(dtype=float)
        out[target_col] = s.to_numpy()

    generators = _normalize_autogluon_generator_names(spec.autogluon_generators)
    if "automl_pipeline" not in generators:
        return out

    try:
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator

        gen = AutoMLPipelineFeatureGenerator(
            enable_numeric_features=True,
            enable_categorical_features=True,
            enable_datetime_features=True,
            enable_text_special_features=False,
            enable_text_ngram_features=False,
            enable_raw_text_features=False,
            enable_vision_features=False,
            verbosity=0,
        )
        transformed = gen.fit_transform(summary_df)
    except Exception:
        transformed = summary_df

    if transformed is None or len(transformed) == 0:
        return out

    transformed = transformed.reset_index(drop=True)
    cols = list(transformed.columns)[: max(1, int(spec.autogluon_max_features))]
    for c in cols:
        vals = pd.to_numeric(transformed[c], errors="coerce")
        if vals.notna().sum() == 0:
            # categorical/object fallback
            vals = pd.Series(pd.factorize(transformed[c].astype(str), sort=True)[0], dtype=float)
            vals = vals.where(vals >= 0, np.nan)
        target_col = f"hist_autogluon_auto_{_sanitize_feature_name(c)}"
        s = pd.Series(np.nan, index=out.index, dtype=float)
        s.iloc[row_idx_fit] = vals.to_numpy(dtype=float)
        out[target_col] = s.to_numpy()
    return out


def _fit_stumpy_matrix_profile(values: np.ndarray, window_size: int, enable_gpu: bool) -> np.ndarray:
    import stumpy

    arr = np.asarray(values, dtype=np.float64)
    m = int(window_size)
    profile = None
    if enable_gpu:
        try:
            profile = stumpy.gpu_stump(arr, m=m)
        except Exception:
            profile = None
    if profile is None:
        profile = stumpy.stump(arr, m=m)

    profile_arr = np.asarray(profile, dtype=object)
    if profile_arr.ndim != 2 or profile_arr.shape[0] == 0:
        return np.empty((0,), dtype=float)

    mp = pd.to_numeric(pd.Series(profile_arr[:, 0]), errors="coerce").to_numpy(dtype=float)
    return mp


def _stumpy_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    shifted = y.shift(1)
    window = int(spec.stumpy_window_size)

    out["hist_stumpy_missing_ratio"] = shifted.isna().rolling(window, min_periods=1).mean().to_numpy()
    out["hist_stumpy_missing_flag"] = shifted.isna().astype(float).to_numpy()
    out["hist_stumpy_mp_score"] = np.nan
    out["hist_stumpy_mp_zscore"] = np.nan
    out["hist_stumpy_discord_flag"] = np.nan

    windows_raw, row_idx = _build_sliding_windows_1d(shifted.to_numpy(dtype=np.float32), window_size=window)
    if windows_raw.shape[0] == 0:
        return out

    min_obs = max(4, window // 4)
    obs_counts = np.isfinite(windows_raw).sum(axis=1)
    keep = obs_counts >= min_obs
    if int(keep.sum()) < int(spec.stumpy_min_train_windows):
        return out

    filled = _fill_series_for_tsfel(shifted, method=spec.stumpy_fill_method)
    arr = filled.to_numpy(dtype=np.float64)
    if arr.size < window + 1:
        return out

    try:
        mp = _fit_stumpy_matrix_profile(arr, window_size=window, enable_gpu=bool(spec.enable_gpu_compute))
    except Exception:
        return out

    n_windows = min(mp.shape[0], keep.shape[0], row_idx.shape[0])
    if n_windows == 0:
        return out

    mp = mp[:n_windows]
    keep = keep[:n_windows]
    row_idx = row_idx[:n_windows]
    valid = keep & np.isfinite(mp)
    if int(valid.sum()) < int(spec.stumpy_min_train_windows):
        return out

    score_s = pd.Series(np.nan, index=y.index, dtype=float)
    score_s.iloc[row_idx[valid]] = mp[valid]
    out["hist_stumpy_mp_score"] = score_s.to_numpy()

    hist_mean = score_s.shift(1).expanding(min_periods=5).mean()
    hist_std = score_s.shift(1).expanding(min_periods=5).std(ddof=0).replace(0.0, np.nan)
    z = (score_s - hist_mean) / (hist_std + 1e-9)
    out["hist_stumpy_mp_zscore"] = z.to_numpy()

    thr = (
        score_s.shift(1)
        .expanding(min_periods=int(spec.stumpy_min_train_windows))
        .quantile(float(spec.stumpy_discord_quantile))
    )
    flag_s = (score_s >= thr).astype(float)
    flag_s[thr.isna() | score_s.isna()] = np.nan
    out["hist_stumpy_discord_flag"] = flag_s.to_numpy()
    return out


def _resolve_tsfresh_default_fc_parameters(feature_set: str) -> Any:
    from tsfresh.feature_extraction.settings import (
        ComprehensiveFCParameters,
        EfficientFCParameters,
        IndexBasedFCParameters,
        MinimalFCParameters,
        TimeBasedFCParameters,
    )

    key = _normalize_tsfresh_feature_set(feature_set)
    if key == "comprehensive":
        return ComprehensiveFCParameters()
    if key == "efficient":
        return EfficientFCParameters()
    if key == "index_based":
        return IndexBasedFCParameters()
    if key == "time_based":
        return TimeBasedFCParameters()
    return MinimalFCParameters()


def _tsfresh_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    shifted = y.shift(1)
    window = int(spec.tsfresh_window_size)

    out["hist_tsfresh_missing_ratio"] = shifted.isna().rolling(window, min_periods=1).mean().to_numpy()
    out["hist_tsfresh_missing_flag"] = shifted.isna().astype(float).to_numpy()

    windows_raw, row_idx = _build_sliding_windows_1d(shifted.to_numpy(dtype=np.float32), window_size=window)
    if windows_raw.shape[0] == 0:
        return out

    min_obs = max(4, window // 4)
    obs_counts = np.isfinite(windows_raw).sum(axis=1)
    keep = obs_counts >= min_obs
    if int(keep.sum()) < int(spec.tsfresh_min_train_windows):
        return out

    filled = _fill_series_for_tsfel(shifted, method=spec.tsfresh_fill_method)
    windows_filled, row_idx_filled = _build_sliding_windows_1d(filled.to_numpy(dtype=np.float32), window_size=window)
    if windows_filled.shape[0] == 0:
        return out

    valid_mask = keep[: min(len(keep), len(windows_filled))]
    windows_fit = windows_filled[: len(valid_mask)][valid_mask]
    row_idx_fit = row_idx_filled[: len(valid_mask)][valid_mask]
    if windows_fit.shape[0] < int(spec.tsfresh_min_train_windows):
        return out

    try:
        from tsfresh.feature_extraction.extraction import extract_features
    except Exception:
        return out

    n_windows, w = windows_fit.shape
    frame = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_windows, dtype=np.int64), w),
            "kind": np.full(n_windows * w, "y", dtype=object),
            "time": np.tile(np.arange(w, dtype=np.int64), n_windows),
            "value": windows_fit.reshape(-1).astype(float),
        }
    )

    try:
        fc_params = _resolve_tsfresh_default_fc_parameters(spec.tsfresh_feature_set)
        feat_df = extract_features(
            frame,
            column_id="id",
            column_kind="kind",
            column_sort="time",
            column_value="value",
            default_fc_parameters=fc_params,
            disable_progressbar=True,
            show_warnings=False,
            n_jobs=int(spec.tsfresh_n_jobs),
            impute_function=None,
        )
    except Exception:
        return out

    if feat_df is None or feat_df.empty:
        return out

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.reindex(index=np.arange(n_windows, dtype=np.int64))
    feat_df = feat_df.reset_index(drop=True)
    cols = list(feat_df.columns)[: max(1, int(spec.tsfresh_max_features))]

    for c in cols:
        raw_name = str(c)
        if "__" in raw_name:
            raw_name = raw_name.split("__", 1)[1]
        target_col = f"hist_tsfresh_{_sanitize_feature_name(raw_name)}"
        if target_col in out.columns:
            continue
        vals = pd.to_numeric(feat_df[c], errors="coerce").to_numpy(dtype=float)
        if vals.shape[0] != row_idx_fit.shape[0]:
            continue
        s = pd.Series(np.nan, index=out.index, dtype=float)
        s.iloc[row_idx_fit] = vals
        out[target_col] = s.to_numpy()
    return out


def _tsfel_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    shifted = y.shift(1)
    window = int(spec.tsfel_window_size)

    out["hist_tsfel_missing_ratio"] = shifted.isna().rolling(window, min_periods=1).mean().to_numpy()
    out["hist_tsfel_missing_flag"] = shifted.isna().astype(float).to_numpy()

    try:
        cfg, feat_names = _build_tsfel_config_and_feature_names(spec)
    except Exception:
        return out

    base_cols = [f"hist_tsfel_{_sanitize_feature_name(n)}" for n in feat_names]
    for c in base_cols:
        out[c] = np.nan

    n = shifted.shape[0]
    n_windows = n - window + 1
    if n_windows < int(spec.tsfel_min_train_windows):
        return out

    filled = _fill_series_for_tsfel(shifted, method=spec.tsfel_fill_method)
    if filled.notna().sum() < window:
        return out

    min_obs = max(4, window // 4)
    row_end_idx_all = np.arange(window - 1, n, dtype=np.int64)
    if row_end_idx_all.size < int(spec.tsfel_min_train_windows):
        return out

    feat_rows: list[dict[str, Any]] = []
    row_end_idx: list[int] = []
    try:
        import tsfel

        for end_idx in row_end_idx_all:
            raw_window = shifted.iloc[end_idx - window + 1 : end_idx + 1]
            if int(raw_window.notna().sum()) < min_obs:
                continue
            win = filled.iloc[end_idx - window + 1 : end_idx + 1]
            one = tsfel.calc_window_features(
                cfg,
                win,
                fs=float(spec.tsfel_sampling_frequency),
                verbose=0,
                single_window=True,
            )
            feat_rows.append(dict(one.iloc[0].to_dict()))
            row_end_idx.append(int(end_idx))
    except Exception:
        return out

    if len(feat_rows) < int(spec.tsfel_min_train_windows):
        return out

    feat_df = pd.DataFrame(feat_rows).reset_index(drop=True)
    row_end_idx_arr = np.asarray(row_end_idx, dtype=np.int64)

    for raw_col in feat_df.columns:
        raw_name = str(raw_col)
        if "_" in raw_name:
            raw_name = raw_name.split("_", 1)[1]
        col = f"hist_tsfel_{_sanitize_feature_name(raw_name)}"
        if col not in out.columns:
            out[col] = np.nan
        values = np.asarray([_coerce_object_to_float(v) for v in feat_df[raw_col].tolist()], dtype=float)
        s = pd.Series(np.nan, index=out.index, dtype=float)
        s.iloc[row_end_idx_arr] = values
        out[col] = s.to_numpy()

    return out


def _anomaly_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    out = g[keys + [spec.time_col]].copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    shifted = y.shift(1)

    hist_mean = shifted.expanding(min_periods=3).mean()
    hist_std = shifted.expanding(min_periods=3).std(ddof=0).replace(0.0, np.nan)
    z = (shifted - hist_mean) / (hist_std + 1e-9)
    out["hist_outlier_zscore_abs"] = z.abs()
    z_flag = (z.abs() > 3.0).astype(float)
    z_flag[z.isna()] = np.nan
    out["hist_outlier_flag_z3"] = z_flag

    win = int(spec.anomaly_rolling_window)
    minp = max(3, min(win, max(5, win // 2)))
    q1 = shifted.rolling(win, min_periods=minp).quantile(0.25)
    q3 = shifted.rolling(win, min_periods=minp).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_score = np.maximum((shifted - upper) / (iqr.abs() + 1e-9), (lower - shifted) / (iqr.abs() + 1e-9))
    iqr_score = np.maximum(iqr_score, 0.0)
    out["hist_outlier_iqr_score"] = iqr_score
    iqr_flag = ((shifted < lower) | (shifted > upper)).astype(float)
    iqr_flag[(q1.isna()) | (q3.isna()) | shifted.isna()] = np.nan
    out["hist_outlier_flag_iqr"] = iqr_flag

    med = shifted.rolling(win, min_periods=minp).median()
    mad = shifted.rolling(win, min_periods=minp).apply(_rolling_mad, raw=True).replace(0.0, np.nan)
    robust_z = 0.6745 * (shifted - med) / (mad + 1e-9)
    out["hist_outlier_robust_z_abs"] = robust_z.abs()
    robust_flag = (robust_z.abs() > 3.5).astype(float)
    robust_flag[robust_z.isna()] = np.nan
    out["hist_outlier_flag_robust"] = robust_flag

    detectors = _normalize_detector_names(spec.pyod_detectors)
    for detector_name in detectors:
        lower_name = detector_name.lower()
        out[f"hist_pyod_{lower_name}_score"] = np.nan
        out[f"hist_pyod_{lower_name}_flag"] = np.nan

    valid = y.notna()
    if valid.sum() < int(spec.anomaly_min_train_size):
        return out

    classes = _load_pyod_detector_classes(detectors)
    x_valid = y[valid].to_numpy(dtype=np.float32).reshape(-1, 1)
    valid_idx = y[valid].index

    for detector_name in detectors:
        detector_class = classes.get(detector_name)
        if detector_class is None:
            continue
        try:
            scores, labels = _fit_pyod_scores_flags(
                detector_class, x_valid, contamination=float(spec.pyod_contamination)
            )
        except Exception:
            continue

        lower_name = detector_name.lower()
        score_s = pd.Series(np.nan, index=y.index, dtype=float)
        score_s.loc[valid_idx] = scores
        flag_s = pd.Series(np.nan, index=y.index, dtype=float)
        flag_s.loc[valid_idx] = labels

        out[f"hist_pyod_{lower_name}_score"] = score_s.shift(1).to_numpy()
        out[f"hist_pyod_{lower_name}_flag"] = flag_s.shift(1).to_numpy()

    return out


def _historical_features_for_group(g: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    base_df = g[keys + [spec.time_col]].reset_index(drop=True).copy()
    y = pd.to_numeric(g[spec.target_col], errors="coerce").astype(float)
    hist_data: dict[str, np.ndarray] = {}

    for lag in [1, 2, 3, 7, 14, 28]:
        hist_data[f"hist_lag_{lag}"] = y.shift(lag).to_numpy()

    hist_data["hist_diff_1"] = y.diff(1).to_numpy()
    hist_data["hist_diff_7"] = y.diff(7).to_numpy()

    shifted = y.shift(1)
    for w in [3, 7, 14, 28]:
        hist_data[f"hist_roll_mean_{w}"] = shifted.rolling(w).mean().to_numpy()
    for w in [7, 14, 28]:
        hist_data[f"hist_roll_std_{w}"] = shifted.rolling(w).std().to_numpy()
    for w in [7, 14]:
        hist_data[f"hist_roll_min_{w}"] = shifted.rolling(w).min().to_numpy()
        hist_data[f"hist_roll_max_{w}"] = shifted.rolling(w).max().to_numpy()

    hist_data["hist_expanding_mean"] = shifted.expanding().mean().to_numpy()
    hist_data["hist_expanding_std"] = shifted.expanding().std().to_numpy()
    hist_data["hist_ewm_mean_7"] = shifted.ewm(span=7, adjust=False).mean().to_numpy()
    hist_data["hist_ewm_mean_14"] = shifted.ewm(span=14, adjust=False).mean().to_numpy()

    frames: list[pd.DataFrame] = [base_df, pd.DataFrame(hist_data, index=base_df.index)]

    if spec.enable_anomaly_features:
        anomaly_df = _anomaly_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(anomaly_df.reset_index(drop=True))

    if spec.enable_merlion_features:
        merlion_df = _merlion_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(merlion_df.reset_index(drop=True))

    if spec.enable_pypots_features:
        pypots_df = _pypots_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(pypots_df.reset_index(drop=True))

    if spec.enable_tsfel_features:
        tsfel_df = _tsfel_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(tsfel_df.reset_index(drop=True))

    if spec.enable_autogluon_features:
        ag_df = _autogluon_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(ag_df.reset_index(drop=True))

    if spec.enable_stumpy_features:
        st_df = _stumpy_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(st_df.reset_index(drop=True))

    if spec.enable_tsfresh_features:
        tsfresh_df = _tsfresh_features_for_group(g, spec).drop(columns=keys + [spec.time_col], errors="ignore")
        frames.append(tsfresh_df.reset_index(drop=True))

    gpu_feats = _gpu_features_for_values(y.values, enabled=spec.enable_gpu_compute)
    if gpu_feats:
        frames.append(pd.DataFrame(gpu_feats, index=base_df.index))

    return pd.concat(frames, axis=1)


def _historical_feature_frame_parallel(df: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    keys = list(spec.group_cols)
    groups = [g for _, g in df.groupby(keys, sort=False)]
    workers = max(1, int(spec.parallel_workers))

    if workers == 1 or len(groups) <= 1:
        group_parts = [_historical_features_for_group(g, spec) for g in groups]
        return pd.concat(group_parts, ignore_index=True)

    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_historical_features_for_group, g, spec) for g in groups]
        for fut in as_completed(futs):
            parts.append(fut.result())
    return pd.concat(parts, ignore_index=True)


def build_exog_dataframe(df: pd.DataFrame, spec: ExogBuildSpec) -> pd.DataFrame:
    _validate_input(df, spec)
    base = _base_feature_frame(df, spec)
    keys = list(spec.group_cols)
    families = set(spec.feature_families)
    effective_spec = spec
    if "anomaly" not in families and spec.enable_anomaly_features:
        effective_spec = dataclasses.replace(spec, enable_anomaly_features=False)

    static_df = _static_feature_frame(base, spec) if "stat" in families else base[keys].drop_duplicates().copy()
    hist_df = (
        _historical_feature_frame_parallel(base, effective_spec) if "hist" in families else base[keys + [spec.time_col]].copy()
    )

    merged = (
        base.merge(static_df, on=keys, how="left")
        .merge(hist_df, on=keys + [spec.time_col], how="left")
        .sort_values(keys + [spec.time_col])
        .reset_index(drop=True)
    )

    base_cols = keys + [spec.time_col, spec.target_col] if "base" in families else []
    prefixed_cols = [c for c in merged.columns if c.startswith(("hist_", "stat_", "feat_"))]
    return merged[base_cols + prefixed_cols]


def _write_exog_table(engine: Engine, df: pd.DataFrame, spec: ExogBuildSpec) -> None:
    schema = safe_ident(spec.target_schema)
    table = safe_ident(spec.target_table)
    keys = [safe_ident(c) for c in spec.group_cols]
    time_col = safe_ident(spec.time_col)

    if spec.postgres_write_mode == "copy":
        copy_dataframe_to_postgres(
            engine,
            df,
            schema=schema,
            table=table,
            if_exists=spec.if_exists,
            chunk_rows=spec.postgres_copy_chunk_rows,
        )
    else:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        df.to_sql(table, engine, schema=schema, if_exists=spec.if_exists, index=False, method="multi", chunksize=10000)

    if spec.create_postgres_index:
        idx_name = safe_ident(f"idx_{table}_key_time")
        cols = ", ".join([f'"{c}"' for c in keys + [time_col]])
        with engine.begin() as conn:
            conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{schema}"."{table}" ({cols})'))


def run_exog_build(spec: ExogBuildSpec) -> dict[str, Any]:
    stage_timing_sec: dict[str, float] = {}
    t0 = time.perf_counter()
    engine = _make_engine(spec)
    run_id = generate_run_id("build_exog")
    monitor = ResourceMonitor(interval_sec=spec.sampling_interval_sec)
    run_status = "failed"
    error_message: str | None = None
    elapsed_started = time.perf_counter()
    source_rows: int | None = None
    written_rows: int | None = None
    feature_cols_count: int | None = None
    target_name = f"{spec.target_schema}.{spec.target_table}"
    source_name = f"{spec.source_schema}.{spec.source_table}"
    docs_summary = _summarize_lib_docs(spec.lib_docs_dir)
    pyod_summary = _summarize_pyod_codegen(spec.pyod_codegen_yaml, spec.pyod_detectors)
    merlion_summary = _summarize_merlion_codegen(spec.merlion_codegen_yaml, spec.merlion_models)
    pypots_summary = _summarize_pypots_codegen(spec.pypots_codegen_yaml, spec.pypots_models)
    tsfel_summary = _summarize_tsfel_codegen(spec.tsfel_codegen_yaml, spec.tsfel_domains, spec.tsfel_max_features)
    autogluon_summary = _summarize_autogluon_codegen(spec.autogluon_codegen_yaml, spec.autogluon_generators)
    stumpy_summary = _summarize_stumpy_codegen(
        spec.stumpy_codegen_yaml, spec.stumpy_window_size, spec.stumpy_discord_quantile
    )
    tsfresh_summary = _summarize_tsfresh_codegen(
        spec.tsfresh_codegen_yaml,
        spec.tsfresh_feature_set,
        spec.tsfresh_max_features,
        spec.tsfresh_n_jobs,
    )
    stage_timing_sec["setup"] = time.perf_counter() - t0
    cfg = ResourcesConfig(
        db_host=spec.host,
        db_port=spec.port,
        db_user=spec.user,
        db_password=spec.password,
        db_name=spec.database,
        schema="resources",
        namespace="timesfm",
        table_naming="plain",
        app_name="loto_exog_builder",
        env=spec.env,
        profile=spec.profile,
        command=f"resources.exog_pipeline {spec.source_schema}.{spec.source_table} -> {spec.target_schema}.{spec.target_table}",
        tags={
            "source": f"{spec.source_schema}.{spec.source_table}",
            "target": f"{spec.target_schema}.{spec.target_table}",
            "group_cols": list(spec.group_cols),
            "gpu_compute": spec.enable_gpu_compute,
            "parallel_workers": spec.parallel_workers,
            "docs_summary": docs_summary,
            "anomaly_features": spec.enable_anomaly_features,
            "pyod_summary": pyod_summary,
            "merlion_features": spec.enable_merlion_features,
            "merlion_summary": merlion_summary,
            "pypots_features": spec.enable_pypots_features,
            "pypots_summary": pypots_summary,
            "tsfel_features": spec.enable_tsfel_features,
            "tsfel_summary": tsfel_summary,
            "autogluon_features": spec.enable_autogluon_features,
            "autogluon_summary": autogluon_summary,
            "stumpy_features": spec.enable_stumpy_features,
            "stumpy_summary": stumpy_summary,
            "tsfresh_features": spec.enable_tsfresh_features,
            "tsfresh_summary": tsfresh_summary,
        },
        enable_gpu=True,
        enable_sampling=True,
        sampling_interval_sec=spec.sampling_interval_sec,
        ensure_schema=True,
        parallel_snapshot_workers=max(1, spec.parallel_workers),
    )

    upsert_model_run(
        engine,
        run_id=run_id,
        model_name="build_exog",
        meta={"command": "build-exog", "source": source_name, "target": target_name},
        library_name="resources",
        adapter_name="build_exog",
        status="running",
    )
    write_log_run_history(
        engine,
        run_id=run_id,
        event_type="build_start",
        status="running",
        model_name="exog",
        library_name="resources",
        adapter_name="build_exog",
        dataset_name=target_name,
        message=f"build-exog started for {target_name}",
        payload={"source_table": source_name, "target_table": target_name},
    )
    monitor.start()
    try:
        with start_run(cfg) as run:
            run.attach_sqlalchemy_engine(engine)

            with run.span(stage_name="analyze_lib_docs", extra={"docs_summary": docs_summary}):
                pass

            t_extract = time.perf_counter()
            with run.span(stage_name="extract_source"):
                src = _read_source(spec, engine)
            source_rows = int(len(src))
            stage_timing_sec["extract"] = time.perf_counter() - t_extract

            t_build = time.perf_counter()
            with run.span(stage_name="build_exog_features", rows_in=int(len(src))):
                out = build_exog_dataframe(src, spec)
            stage_timing_sec["build"] = time.perf_counter() - t_build

            t_write = time.perf_counter()
            with run.span(stage_name="write_exog_table", rows_in=int(len(out)), rows_out=int(len(out))):
                _write_exog_table(engine, out, spec)
            stage_timing_sec["write"] = time.perf_counter() - t_write

            run.set_counts(
                rows_target=int(len(src)), rows_written=int(len(out)), rows_failed=max(0, int(len(src) - len(out)))
            )
            written_rows = int(len(out))

            feature_cols = [
                c for c in out.columns if c.startswith("hist_") or c.startswith("stat_") or c.startswith("feat_")
            ]
            feature_cols_count = len(feature_cols)
            elapsed_sec = time.perf_counter() - elapsed_started
            write_log_run_history(
                engine,
                run_id=run_id,
                event_type="build_end",
                status="success",
                model_name="exog",
                library_name="resources",
                adapter_name="build_exog",
                dataset_name=target_name,
                message=f"build-exog completed for {target_name}",
                payload={
                    "source_table": source_name,
                    "target_table": target_name,
                    "source_rows": source_rows,
                    "written_rows": written_rows,
                    "feature_cols": feature_cols_count,
                    "elapsed_sec": elapsed_sec,
                    "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
                },
            )

            return {
                "run_id": run_id,
                "source_rows": int(len(src)),
                "written_rows": int(len(out)),
                "feature_cols": len(feature_cols),
                "target": f"{spec.target_schema}.{spec.target_table}",
                "docs_summary": docs_summary,
                "pyod_summary": pyod_summary,
                "merlion_summary": merlion_summary,
                "pypots_summary": pypots_summary,
                "tsfel_summary": tsfel_summary,
                "autogluon_summary": autogluon_summary,
                "stumpy_summary": stumpy_summary,
                "tsfresh_summary": tsfresh_summary,
                "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
            }
    except Exception as exc:
        error_message = str(exc)
        elapsed_sec = time.perf_counter() - elapsed_started
        write_log_run_history(
            engine,
            run_id=run_id,
            event_type="build_end",
            status="failed",
            model_name="exog",
            library_name="resources",
            adapter_name="build_exog",
            dataset_name=target_name,
            message=f"build-exog failed for {target_name}",
            payload={
                "source_table": source_name,
                "target_table": target_name,
                "source_rows": source_rows,
                "written_rows": written_rows,
                "elapsed_sec": elapsed_sec,
                "stage_timing_sec": stage_timing_sec if spec.profile_stages else None,
                "error_message": error_message,
            },
        )
        raise
    finally:
        samples = monitor.stop()
        if samples:
            write_resource_samples(engine, run_id=run_id, samples=monitor.to_dicts())
        if error_message is None:
            run_status = "success"
        mark_model_run_end(engine, run_id=run_id, status=run_status, error_message=error_message)


def _parse_group_cols(raw: str) -> tuple[str, ...]:
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    if not cols:
        raise ValueError("group-cols must not be empty")
    return tuple(cols)


def _parse_detector_csv(raw: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return _normalize_detector_names(tuple(vals))


def _parse_merlion_model_csv(raw: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return _normalize_merlion_model_names(tuple(vals))


def _parse_pypots_model_csv(raw: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return _normalize_pypots_model_names(tuple(vals))


def _parse_tsfel_domains_csv(raw: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return _normalize_tsfel_domain_names(tuple(vals))


def _parse_autogluon_generators_csv(raw: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return _normalize_autogluon_generator_names(tuple(vals))


def _parse_tsfresh_feature_set(raw: str) -> str:
    return _normalize_tsfresh_feature_set(str(raw))


def _parse_feature_families(raw: str) -> tuple[str, ...]:
    vals = tuple(x.strip() for x in str(raw).split(",") if x.strip())
    if not vals:
        return ("base", "hist", "stat")
    allowed = {"base", "hist", "stat", "anomaly"}
    invalid = [v for v in vals if v not in allowed]
    if invalid:
        raise ValueError(f"unsupported feature families: {invalid}")
    return vals


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m resources.exog_pipeline")
    p.add_argument("--profile", default="local")
    p.add_argument("--env", default="LOCAL")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5432)
    p.add_argument("--user", default="loto")
    p.set_defaults(password=os.environ.get("DB_PASSWORD", ""))
    p.add_argument("--database", default="loto")

    p.add_argument("--source-schema", default="dataset")
    p.add_argument("--source-table", default="loto_y_ts")
    p.add_argument("--source-where", default=None)

    p.add_argument("--target-schema", default="exog")
    p.add_argument("--target-table", default="loto_y_ts_exog")
    p.add_argument("--if-exists", choices=["replace", "append", "fail"], default="replace")

    p.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p.add_argument("--time-col", default="ds")
    p.add_argument("--target-col", default="y")

    p.add_argument("--parallel-workers", type=int, default=4)
    p.add_argument("--profile-stages", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--row-batch-size", type=int, default=5000)
    p.add_argument("--max-groups-per-batch", type=int, default=64)
    p.add_argument("--feature-families", default="base,hist,stat")
    p.add_argument("--enable-gpu-compute", action="store_true")
    p.add_argument("--enable-anomaly-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--postgres-write-mode", default="copy", choices=["copy", "to_sql"])
    p.add_argument("--postgres-copy-chunk-rows", type=int, default=50000)
    p.add_argument("--create-postgres-index", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pyod-codegen-yaml", default="./docs/lib_docs/pyod_all_codegen.yaml")
    p.add_argument("--pyod-detectors", default="ECOD,IForest,COPOD")
    p.add_argument("--pyod-contamination", type=float, default=0.1)
    p.add_argument("--anomaly-min-train-size", type=int, default=20)
    p.add_argument("--anomaly-rolling-window", type=int, default=14)
    p.add_argument("--enable-merlion-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--merlion-codegen-yaml",
        default="./docs/lib_docs/merlion_dashboard_selected_codegen_details.yaml",
    )
    p.add_argument("--merlion-models", default="iforest,lof,spectral_residual,stat_threshold")
    p.add_argument("--merlion-contamination", type=float, default=0.1)
    p.add_argument("--merlion-min-train-size", type=int, default=30)
    p.add_argument("--merlion-n-estimators", type=int, default=100)
    p.add_argument("--merlion-max-n-samples", type=int, default=512)
    p.add_argument("--merlion-random-state", type=int, default=42)
    p.add_argument("--enable-pypots-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pypots-codegen-yaml", default="./docs/lib_docs/pypots_all_codegen.yaml")
    p.add_argument("--pypots-models", default="transformer,saits")
    p.add_argument("--pypots-anomaly-rate", type=float, default=0.1)
    p.add_argument("--pypots-window-size", type=int, default=32)
    p.add_argument("--pypots-min-train-windows", type=int, default=20)
    p.add_argument("--pypots-epochs", type=int, default=2)
    p.add_argument("--pypots-batch-size", type=int, default=32)
    p.add_argument("--enable-tsfel-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tsfel-codegen-yaml", default="./docs/lib_docs/tsfel_all_codegen.yaml")
    p.add_argument("--tsfel-domains", default="statistical,temporal,spectral")
    p.add_argument("--tsfel-max-features", type=int, default=64)
    p.add_argument("--tsfel-window-size", type=int, default=32)
    p.add_argument("--tsfel-min-train-windows", type=int, default=20)
    p.add_argument("--tsfel-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill")
    p.add_argument("--tsfel-sampling-frequency", type=float, default=1.0)
    p.add_argument("--enable-autogluon-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--autogluon-codegen-yaml", default="./docs/lib_docs/autogluon__internal__all_codegen.yaml"
    )
    p.add_argument("--autogluon-generators", default="automl_pipeline")
    p.add_argument("--autogluon-window-size", type=int, default=32)
    p.add_argument("--autogluon-min-train-windows", type=int, default=20)
    p.add_argument(
        "--autogluon-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill"
    )
    p.add_argument("--autogluon-max-features", type=int, default=64)
    p.add_argument("--enable-stumpy-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--stumpy-codegen-yaml", default="./docs/lib_docs/stumpy_all_codegen.yaml")
    p.add_argument("--stumpy-window-size", type=int, default=32)
    p.add_argument("--stumpy-min-train-windows", type=int, default=20)
    p.add_argument("--stumpy-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill")
    p.add_argument("--stumpy-discord-quantile", type=float, default=0.98)
    p.add_argument("--enable-tsfresh-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tsfresh-codegen-yaml", default="./docs/lib_docs/tsfresh_all_codegen.yaml")
    p.add_argument("--tsfresh-feature-set", default="minimal")
    p.add_argument("--tsfresh-window-size", type=int, default=32)
    p.add_argument("--tsfresh-min-train-windows", type=int, default=20)
    p.add_argument("--tsfresh-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill")
    p.add_argument("--tsfresh-max-features", type=int, default=64)
    p.add_argument("--tsfresh-n-jobs", type=int, default=0)
    p.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p.add_argument("--lib-docs-dir", default="./docs/lib_docs")

    args = p.parse_args()
    spec = ExogBuildSpec(
        profile=args.profile,
        env=args.env,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        source_schema=args.source_schema,
        source_table=args.source_table,
        source_where=args.source_where,
        target_schema=args.target_schema,
        target_table=args.target_table,
        if_exists=args.if_exists,
        group_cols=_parse_group_cols(args.group_cols),
        time_col=args.time_col,
        target_col=args.target_col,
        parallel_workers=args.parallel_workers,
        profile_stages=bool(args.profile_stages),
        row_batch_size=int(args.row_batch_size),
        max_groups_per_batch=int(args.max_groups_per_batch),
        feature_families=_parse_feature_families(args.feature_families),
        enable_gpu_compute=bool(args.enable_gpu_compute),
        enable_anomaly_features=bool(args.enable_anomaly_features),
        create_postgres_index=bool(args.create_postgres_index),
        postgres_write_mode=str(args.postgres_write_mode),
        postgres_copy_chunk_rows=int(args.postgres_copy_chunk_rows),
        pyod_codegen_yaml=args.pyod_codegen_yaml,
        pyod_detectors=_parse_detector_csv(args.pyod_detectors),
        pyod_contamination=float(args.pyod_contamination),
        anomaly_min_train_size=int(args.anomaly_min_train_size),
        anomaly_rolling_window=int(args.anomaly_rolling_window),
        enable_merlion_features=bool(args.enable_merlion_features),
        merlion_codegen_yaml=args.merlion_codegen_yaml,
        merlion_models=_parse_merlion_model_csv(args.merlion_models),
        merlion_contamination=float(args.merlion_contamination),
        merlion_min_train_size=int(args.merlion_min_train_size),
        merlion_n_estimators=int(args.merlion_n_estimators),
        merlion_max_n_samples=int(args.merlion_max_n_samples),
        merlion_random_state=int(args.merlion_random_state),
        enable_pypots_features=bool(args.enable_pypots_features),
        pypots_codegen_yaml=args.pypots_codegen_yaml,
        pypots_models=_parse_pypots_model_csv(args.pypots_models),
        pypots_anomaly_rate=float(args.pypots_anomaly_rate),
        pypots_window_size=int(args.pypots_window_size),
        pypots_min_train_windows=int(args.pypots_min_train_windows),
        pypots_epochs=int(args.pypots_epochs),
        pypots_batch_size=int(args.pypots_batch_size),
        enable_tsfel_features=bool(args.enable_tsfel_features),
        tsfel_codegen_yaml=args.tsfel_codegen_yaml,
        tsfel_domains=_parse_tsfel_domains_csv(args.tsfel_domains),
        tsfel_max_features=int(args.tsfel_max_features),
        tsfel_window_size=int(args.tsfel_window_size),
        tsfel_min_train_windows=int(args.tsfel_min_train_windows),
        tsfel_fill_method=str(args.tsfel_fill_method),
        tsfel_sampling_frequency=float(args.tsfel_sampling_frequency),
        enable_autogluon_features=bool(args.enable_autogluon_features),
        autogluon_codegen_yaml=args.autogluon_codegen_yaml,
        autogluon_generators=_parse_autogluon_generators_csv(args.autogluon_generators),
        autogluon_window_size=int(args.autogluon_window_size),
        autogluon_min_train_windows=int(args.autogluon_min_train_windows),
        autogluon_fill_method=str(args.autogluon_fill_method),
        autogluon_max_features=int(args.autogluon_max_features),
        enable_stumpy_features=bool(args.enable_stumpy_features),
        stumpy_codegen_yaml=args.stumpy_codegen_yaml,
        stumpy_window_size=int(args.stumpy_window_size),
        stumpy_min_train_windows=int(args.stumpy_min_train_windows),
        stumpy_fill_method=str(args.stumpy_fill_method),
        stumpy_discord_quantile=float(args.stumpy_discord_quantile),
        enable_tsfresh_features=bool(args.enable_tsfresh_features),
        tsfresh_codegen_yaml=args.tsfresh_codegen_yaml,
        tsfresh_feature_set=_parse_tsfresh_feature_set(args.tsfresh_feature_set),
        tsfresh_window_size=int(args.tsfresh_window_size),
        tsfresh_min_train_windows=int(args.tsfresh_min_train_windows),
        tsfresh_fill_method=str(args.tsfresh_fill_method),
        tsfresh_max_features=int(args.tsfresh_max_features),
        tsfresh_n_jobs=int(args.tsfresh_n_jobs),
        sampling_interval_sec=args.sampling_interval_sec,
        lib_docs_dir=args.lib_docs_dir,
    )
    out = run_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
