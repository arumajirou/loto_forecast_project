from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..analysis.explain import permutation_importance_exog
from ..config.settings import settings
from ..data.db import execute_sql_file, make_engine
from ..data.unified_dataset import UnifiedBuildSpec, build_unified_dataset, persist_unified_outputs
from ..models.registry import get_adapter
from .grid_runner import expand_param_grid
from .pipeline import evaluate, predict_with_dataset, train


def _safe_ident(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch == "_")
    if not cleaned:
        raise ValueError(f"invalid identifier: {value}")
    return cleaned


def _to_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        return dict(json.loads(value))
    return {}


def _to_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        loaded = json.loads(value)
        return list(loaded) if isinstance(loaded, list) else []
    return []


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


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _stable_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _build_combo_signature(model_name: str, horizon: int, params_obj: dict[str, Any]) -> str:
    payload = {
        "model_name": str(model_name or "").strip(),
        "horizon": max(1, int(horizon or 1)),
        "params_json": dict(params_obj or {}),
    }
    raw = _stable_json_dumps(payload)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _load_success_combo_signatures(engine: Engine) -> set[str]:
    schema = _safe_ident(settings.model_schema)
    table = _safe_ident(settings.model_table)
    q_primary = text(
        f"""
        SELECT model_name, horizon, params_json, diagnostics_json
        FROM {schema}.{table}
        WHERE LOWER(COALESCE(status, '')) = 'success'
          AND NULLIF(TRIM(COALESCE(error_message, '')), '') IS NULL
        """
    )
    q_fallback = text(
        f"""
        SELECT model_name, horizon, params_json, diagnostics_json
        FROM {schema}.{table}
        WHERE LOWER(COALESCE(status, '')) = 'success'
        """
    )
    try:
        with engine.begin() as conn:
            rows = conn.execute(q_primary).mappings().all()
    except Exception:
        try:
            with engine.begin() as conn:
                rows = conn.execute(q_fallback).mappings().all()
        except Exception:
            rows = []
    out: set[str] = set()
    for row in rows:
        diag = _to_json_dict(row.get("diagnostics_json"))
        sig = str(diag.get("combo_signature") or "").strip().lower()
        if sig:
            out.add(sig)
        params = _to_json_dict(row.get("params_json"))
        try:
            fallback_sig = _build_combo_signature(
                str(row.get("model_name") or ""),
                int(row.get("horizon") or 1),
                params,
            )
            out.add(fallback_sig.lower())
        except Exception:
            pass
    return out


def _normalize_group_cols(value: Any) -> list[str]:
    raw = _to_json_list(value)
    cols = [str(x).strip() for x in raw if str(x).strip()]
    if not cols:
        cols = ["loto", "unique_id", "ts_type"]
    return list(dict.fromkeys(cols))


def _dedupe_json_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for v in values:
        try:
            key = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _normalize_param_mode_json(value: Any) -> dict[str, dict[str, Any]]:
    raw = _to_json_dict(value)
    out: dict[str, dict[str, Any]] = {}
    for key, spec_raw in raw.items():
        name = str(key).strip()
        if not name:
            continue
        mode = "fixed"
        enabled = True
        value_set = False
        value_payload: Any = None
        values_payload: list[Any] | None = None
        if isinstance(spec_raw, str):
            mode = str(spec_raw).strip().lower()
        elif isinstance(spec_raw, bool):
            mode = "vary" if bool(spec_raw) else "fixed"
        elif isinstance(spec_raw, dict):
            mode = str(spec_raw.get("mode") or "").strip().lower()
            if not mode:
                mode = "vary" if _to_bool(spec_raw.get("variable"), False) else "fixed"
            enabled = _to_bool(spec_raw.get("enabled"), True)
            if "value" in spec_raw:
                value_set = True
                value_payload = spec_raw.get("value")
            if "values" in spec_raw:
                raw_values = spec_raw.get("values")
                if isinstance(raw_values, list):
                    values_payload = _dedupe_json_values(list(raw_values))
                else:
                    values_payload = [raw_values]
        else:
            continue
        if mode in {"variable", "vary", "grid", "search"}:
            mode = "vary"
        elif mode in {"fixed", "const", "constant"}:
            mode = "fixed"
        else:
            mode = "fixed"
        normalized = {"mode": mode, "enabled": bool(enabled)}
        if value_set:
            normalized["value"] = value_payload
        if values_payload is not None:
            normalized["values"] = values_payload
        out[name] = normalized
    return out


def _apply_param_mode(
    model_params: dict[str, Any] | None,
    param_space: dict[str, Any] | None,
    param_mode: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, list[Any]], dict[str, dict[str, Any]]]:
    fixed = dict(model_params or {})
    space_raw = dict(param_space or {})
    space: dict[str, list[Any]] = {}
    for key, value in space_raw.items():
        name = str(key)
        if isinstance(value, list):
            vals = _dedupe_json_values(list(value))
            if vals:
                space[name] = vals
            continue
        space[name] = [value]

    mode_map = _normalize_param_mode_json(param_mode)
    for key, spec in mode_map.items():
        enabled = bool(spec.get("enabled", True))
        mode = str(spec.get("mode") or "fixed").strip().lower()
        if not enabled:
            fixed.pop(key, None)
            space.pop(key, None)
            continue
        if mode == "vary":
            values = spec.get("values")
            if isinstance(values, list) and values:
                vals = _dedupe_json_values(list(values))
            elif "value" in spec:
                vals = [spec.get("value")]
            elif key in space and isinstance(space.get(key), list) and space.get(key):
                vals = list(space.get(key) or [])
            elif key in fixed:
                vals = [fixed.get(key)]
            else:
                vals = []
            if vals:
                space[key] = _dedupe_json_values(vals)
            fixed.pop(key, None)
        else:
            if "value" in spec:
                fixed[key] = spec.get("value")
            elif key not in fixed and key in space and isinstance(space.get(key), list) and space.get(key):
                fixed[key] = list(space.get(key) or [None])[0]
            space.pop(key, None)
    return fixed, space, mode_map


def _meta_rows(engine: Engine, config_id: int | None = None, limit: int = 100) -> list[dict[str, Any]]:
    schema = _safe_ident(settings.meta_schema)
    table = _safe_ident(settings.meta_table)
    if config_id is None:
        q = text(
            f"""
            SELECT *
            FROM {schema}.{table}
            WHERE active = TRUE
            ORDER BY priority, config_id
            LIMIT :limit
            """
        )
        params = {"limit": int(limit)}
    else:
        q = text(
            f"""
            SELECT *
            FROM {schema}.{table}
            WHERE config_id = :config_id
            LIMIT 1
            """
        )
        params = {"config_id": int(config_id)}

    with engine.connect() as conn:
        rows = conn.execute(q, params).mappings().all()
    return [dict(r) for r in rows]


def _normalize_meta_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    config_name = str(config.get("config_name") or "").strip()
    if not config_name:
        raise ValueError("config_name is required")

    raw_auto_config = _to_json_dict(config.get("auto_config_json"))
    raw_model_params = _to_json_dict(config.get("model_params_json"))
    raw_param_space = _to_json_dict(config.get("param_space_json"))
    raw_param_mode = _to_json_dict(config.get("param_mode_json"))
    normalized_model_params, normalized_param_space, normalized_param_mode = _apply_param_mode(
        raw_model_params,
        raw_param_space,
        raw_param_mode,
    )

    return {
        "active": _to_bool(config.get("active"), True),
        "priority": _to_int(config.get("priority"), 100),
        "config_name": config_name,
        "base_schema": str(config.get("base_schema") or "dataset"),
        "base_table": str(config.get("base_table") or "loto_y_ts"),
        "hist_schema": str(config.get("hist_schema") or "dataset"),
        "hist_table": str(config.get("hist_table") or "loto_hist_feat"),
        "exog_schema": str(config.get("exog_schema") or "exog"),
        "output_schema": str(config.get("output_schema") or settings.exog_schema),
        "output_table": str(config.get("output_table") or "loto_y_ts_unified"),
        "output_if_exists": str(config.get("output_if_exists") or "replace"),
        "output_csv_path": config.get("output_csv_path"),
        "output_parquet_path": config.get("output_parquet_path"),
        "output_spark_path": config.get("output_spark_path"),
        "output_spark_format": str(config.get("output_spark_format") or "parquet"),
        "unified_filter_json": _to_json_dict(config.get("unified_filter_json")),
        "unified_group_cols_json": _normalize_group_cols(config.get("unified_group_cols_json")),
        "unified_group_validate_strict": _to_bool(config.get("unified_group_validate_strict"), False),
        "model_name": str(config.get("model_name") or "AutoNHITS"),
        "horizon": _to_int(config.get("horizon"), settings.default_horizon),
        "auto_cls_model": (
            str(config.get("auto_cls_model")).strip() if config.get("auto_cls_model") is not None else None
        )
        or None,
        "auto_h": (
            max(1, _to_int(config.get("auto_h"), 1))
            if (config.get("auto_h") is not None and str(config.get("auto_h")).strip() != "")
            else None
        ),
        "auto_loss": str(config.get("auto_loss") or "MAE"),
        "auto_valid_loss": str(config.get("auto_valid_loss") or "MAE"),
        "auto_config_json": raw_auto_config,
        "auto_search_alg": str(config.get("auto_search_alg") or "BasicVariantGenerator"),
        "auto_num_samples": max(1, _to_int(config.get("auto_num_samples"), 10)),
        "auto_cpus": config.get("auto_cpus"),
        "auto_gpus": config.get("auto_gpus"),
        "auto_refit_with_val": _to_bool(config.get("auto_refit_with_val"), False),
        "auto_verbose": _to_bool(config.get("auto_verbose"), False),
        "auto_alias": config.get("auto_alias"),
        "auto_backend": str(config.get("auto_backend") or "ray"),
        "auto_callbacks_json": _to_json_list(config.get("auto_callbacks_json")),
        "model_params_json": normalized_model_params,
        "param_space_json": normalized_param_space,
        "param_mode_json": normalized_param_mode,
        "random_seed": _to_int(config.get("random_seed"), 1),
        "max_tasks": config.get("max_tasks"),
        "recursive_depth": max(1, _to_int(config.get("recursive_depth"), 1)),
        "run_predict": _to_bool(config.get("run_predict"), True),
        "run_evaluate": _to_bool(config.get("run_evaluate"), True),
        "run_explain": _to_bool(config.get("run_explain"), True),
        "run_save": _to_bool(config.get("run_save"), True),
        "run_load": _to_bool(config.get("run_load"), True),
        "run_analyze": _to_bool(config.get("run_analyze"), True),
        "explain_repeats": max(1, _to_int(config.get("explain_repeats"), 3)),
        "save_dataset": _to_bool(config.get("save_dataset"), False),
        "save_overwrite": _to_bool(config.get("save_overwrite"), True),
        "save_path": config.get("save_path"),
        "load_check_predict": _to_bool(config.get("load_check_predict"), False),
        "note": config.get("note"),
    }


def _validate_param_space_shape(param_space: dict[str, Any]) -> tuple[list[str], dict[str, list[Any]]]:
    errors: list[str] = []
    normalized: dict[str, list[Any]] = {}
    for key, value in (param_space or {}).items():
        name = str(key)
        if not isinstance(value, list):
            errors.append(f"param_space_json[{name}] must be list")
            continue
        if len(value) == 0:
            errors.append(f"param_space_json[{name}] must not be empty")
            continue
        normalized[name] = list(value)
    return errors, normalized


def _validate_meta_model_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = str(payload.get("auto_cls_model") or payload.get("model_name") or "AutoNHITS")
    adapter = get_adapter("neuralforecast_auto")
    errors: list[str] = []
    warnings: list[str] = []

    model_params = _to_json_dict(payload.get("model_params_json"))
    auto_config = _to_json_dict(payload.get("auto_config_json"))
    param_space = _to_json_dict(payload.get("param_space_json"))
    param_mode = _to_json_dict(payload.get("param_mode_json"))
    merged_for_mode = dict(auto_config)
    merged_for_mode.update(model_params)
    model_params, param_space, param_mode = _apply_param_mode(
        merged_for_mode,
        param_space,
        param_mode,
    )

    base_validation = adapter.validate(model_name=model_name, model_params={})
    accepted_params = set(base_validation.get("accepted_params") or [])

    v_model = adapter.validate(model_name=model_name, model_params=model_params)
    if not v_model.get("ok", False):
        errors.extend([f"model_params_json: {e}" for e in v_model.get("errors", [])])
    warnings.extend([f"model_params_json: {w}" for w in v_model.get("warnings", [])])

    v_auto = adapter.validate(model_name=model_name, model_params=auto_config)
    if not v_auto.get("ok", False):
        errors.extend([f"auto_config_json: {e}" for e in v_auto.get("errors", [])])
    warnings.extend([f"auto_config_json: {w}" for w in v_auto.get("warnings", [])])

    ps_shape_errors, ps_norm = _validate_param_space_shape(param_space)
    errors.extend(ps_shape_errors)
    for key, values in ps_norm.items():
        if accepted_params and key not in accepted_params:
            errors.append(f"param_space_json: unknown param key={key}")
            continue
        for idx, item in enumerate(values, start=1):
            v_item = adapter.validate(model_name=model_name, model_params={key: item})
            if not v_item.get("ok", False):
                item_errors = ", ".join(v_item.get("errors", []))
                errors.append(f"param_space_json[{key}][{idx}]: {item_errors}")

    merged = dict(auto_config)
    merged.update(model_params)
    v_merged = adapter.validate(model_name=model_name, model_params=merged)
    if not v_merged.get("ok", False):
        errors.extend([f"merged_params: {e}" for e in v_merged.get("errors", [])])
    warnings.extend([f"merged_params: {w}" for w in v_merged.get("warnings", [])])

    return {
        "ok": len(errors) == 0,
        "model_name": model_name,
        "errors": errors,
        "warnings": warnings,
        "accepted_params": v_merged.get("accepted_params", []),
        "required_model_params": v_merged.get("required_model_params", []),
        "reserved_param_specs": v_merged.get("reserved_param_specs", {}),
        "sections": {
            "model_params_json": model_params,
            "auto_config_json": auto_config,
            "param_space_json": ps_norm,
            "param_mode_json": param_mode,
        },
    }


def create_meta_automodel_config(config: dict[str, Any], upsert_by_name: bool = True) -> dict[str, Any]:
    engine = make_engine()
    payload = _normalize_meta_config(config)
    arg_validation = _validate_meta_model_arguments(payload)
    if not arg_validation.get("ok", False):
        detail = "; ".join([str(e) for e in arg_validation.get("errors", [])])
        raise ValueError(f"meta argument validation failed: {detail}")
    schema = _safe_ident(settings.meta_schema)
    table = _safe_ident(settings.meta_table)
    q_find = text(
        f"""
        SELECT config_id
        FROM {schema}.{table}
        WHERE config_name = :config_name
        ORDER BY config_id
        LIMIT 1
        """
    )
    q_insert = text(
        f"""
        INSERT INTO {schema}.{table} (
          active, priority, config_name,
          base_schema, base_table, hist_schema, hist_table, exog_schema,
          output_schema, output_table, output_if_exists,
          output_csv_path, output_parquet_path, output_spark_path, output_spark_format,
          unified_filter_json,
          unified_group_cols_json, unified_group_validate_strict,
          model_name, horizon,
          auto_cls_model, auto_h, auto_loss, auto_valid_loss, auto_config_json, auto_search_alg, auto_num_samples,
          auto_cpus, auto_gpus, auto_refit_with_val, auto_verbose, auto_alias, auto_backend, auto_callbacks_json,
          model_params_json, param_space_json, param_mode_json,
          random_seed, max_tasks, recursive_depth,
          run_predict, run_evaluate, run_explain, run_save, run_load, run_analyze, explain_repeats,
          save_dataset, save_overwrite, save_path, load_check_predict,
          note
        ) VALUES (
          :active, :priority, :config_name,
          :base_schema, :base_table, :hist_schema, :hist_table, :exog_schema,
          :output_schema, :output_table, :output_if_exists,
          :output_csv_path, :output_parquet_path, :output_spark_path, :output_spark_format,
          CAST(:unified_filter_json AS jsonb),
          CAST(:unified_group_cols_json AS jsonb), :unified_group_validate_strict,
          :model_name, :horizon,
          :auto_cls_model, :auto_h, :auto_loss, :auto_valid_loss, CAST(:auto_config_json AS jsonb), :auto_search_alg, :auto_num_samples,
          :auto_cpus, :auto_gpus, :auto_refit_with_val, :auto_verbose, :auto_alias, :auto_backend, CAST(:auto_callbacks_json AS jsonb),
          CAST(:model_params_json AS jsonb), CAST(:param_space_json AS jsonb), CAST(:param_mode_json AS jsonb),
          :random_seed, :max_tasks, :recursive_depth,
          :run_predict, :run_evaluate, :run_explain, :run_save, :run_load, :run_analyze, :explain_repeats,
          :save_dataset, :save_overwrite, :save_path, :load_check_predict,
          :note
        )
        RETURNING config_id
        """
    )
    q_update = text(
        f"""
        UPDATE {schema}.{table}
        SET
          active = :active,
          priority = :priority,
          config_name = :config_name,
          base_schema = :base_schema,
          base_table = :base_table,
          hist_schema = :hist_schema,
          hist_table = :hist_table,
          exog_schema = :exog_schema,
          output_schema = :output_schema,
          output_table = :output_table,
          output_if_exists = :output_if_exists,
          output_csv_path = :output_csv_path,
          output_parquet_path = :output_parquet_path,
          output_spark_path = :output_spark_path,
          output_spark_format = :output_spark_format,
          unified_filter_json = CAST(:unified_filter_json AS jsonb),
          unified_group_cols_json = CAST(:unified_group_cols_json AS jsonb),
          unified_group_validate_strict = :unified_group_validate_strict,
          model_name = :model_name,
          horizon = :horizon,
          auto_cls_model = :auto_cls_model,
          auto_h = :auto_h,
          auto_loss = :auto_loss,
          auto_valid_loss = :auto_valid_loss,
          auto_config_json = CAST(:auto_config_json AS jsonb),
          auto_search_alg = :auto_search_alg,
          auto_num_samples = :auto_num_samples,
          auto_cpus = :auto_cpus,
          auto_gpus = :auto_gpus,
          auto_refit_with_val = :auto_refit_with_val,
          auto_verbose = :auto_verbose,
          auto_alias = :auto_alias,
          auto_backend = :auto_backend,
          auto_callbacks_json = CAST(:auto_callbacks_json AS jsonb),
          model_params_json = CAST(:model_params_json AS jsonb),
          param_space_json = CAST(:param_space_json AS jsonb),
          param_mode_json = CAST(:param_mode_json AS jsonb),
          random_seed = :random_seed,
          max_tasks = :max_tasks,
          recursive_depth = :recursive_depth,
          run_predict = :run_predict,
          run_evaluate = :run_evaluate,
          run_explain = :run_explain,
          run_save = :run_save,
          run_load = :run_load,
          run_analyze = :run_analyze,
          explain_repeats = :explain_repeats,
          save_dataset = :save_dataset,
          save_overwrite = :save_overwrite,
          save_path = :save_path,
          load_check_predict = :load_check_predict,
          note = :note,
          updated_at = now()
        WHERE config_id = :config_id
        """
    )
    params = dict(payload)
    params["unified_filter_json"] = json.dumps(payload["unified_filter_json"], ensure_ascii=False)
    params["unified_group_cols_json"] = json.dumps(payload["unified_group_cols_json"], ensure_ascii=False)
    params["auto_config_json"] = json.dumps(payload["auto_config_json"], ensure_ascii=False)
    params["auto_callbacks_json"] = json.dumps(payload["auto_callbacks_json"], ensure_ascii=False)
    params["model_params_json"] = json.dumps(payload["model_params_json"], ensure_ascii=False)
    params["param_space_json"] = json.dumps(payload["param_space_json"], ensure_ascii=False)
    params["param_mode_json"] = json.dumps(payload["param_mode_json"], ensure_ascii=False)

    action = "inserted"
    config_id: int
    with engine.begin() as conn:
        existing = None
        if upsert_by_name:
            existing = conn.execute(q_find, {"config_name": payload["config_name"]}).mappings().first()
        if existing is None:
            row = conn.execute(q_insert, params).mappings().first()
            if row is None:
                raise RuntimeError("meta automodel insert did not return config_id")
            config_id = int(row["config_id"])
        else:
            action = "updated"
            config_id = int(existing["config_id"])
            update_params = dict(params)
            update_params["config_id"] = config_id
            conn.execute(q_update, update_params)

    rows = _meta_rows(engine, config_id=config_id, limit=1)
    return {
        "action": action,
        "config_id": config_id,
        "meta_schema": schema,
        "meta_table": table,
        "config": rows[0] if rows else {},
        "argument_validation": arg_validation,
    }


def _write_result(engine: Engine, payload: dict[str, Any]) -> None:
    schema = _safe_ident(settings.model_schema)
    table = _safe_ident(settings.model_table)
    q = text(
        f"""
        INSERT INTO {schema}.{table} (
          config_id, run_id, status, model_name, horizon,
          params_json, exog_json, metrics_json, diagnostics_json, explain_json,
          model_save_json, model_load_json, model_analyze_json, model_store_path,
          artifact_path, log_path, unified_schema, unified_table,
          dataset_rows, feature_cols, error_message, started_at, ended_at
        ) VALUES (
          :config_id, :run_id, :status, :model_name, :horizon,
          CAST(:params_json AS jsonb), CAST(:exog_json AS jsonb), CAST(:metrics_json AS jsonb),
          CAST(:diagnostics_json AS jsonb), CAST(:explain_json AS jsonb),
          CAST(:model_save_json AS jsonb), CAST(:model_load_json AS jsonb), CAST(:model_analyze_json AS jsonb), :model_store_path,
          :artifact_path, :log_path, :unified_schema, :unified_table,
          :dataset_rows, :feature_cols, :error_message, :started_at, :ended_at
        )
        ON CONFLICT (run_id) DO UPDATE SET
          status = EXCLUDED.status,
          model_name = EXCLUDED.model_name,
          horizon = EXCLUDED.horizon,
          params_json = EXCLUDED.params_json,
          exog_json = EXCLUDED.exog_json,
          metrics_json = EXCLUDED.metrics_json,
          diagnostics_json = EXCLUDED.diagnostics_json,
          explain_json = EXCLUDED.explain_json,
          model_save_json = EXCLUDED.model_save_json,
          model_load_json = EXCLUDED.model_load_json,
          model_analyze_json = EXCLUDED.model_analyze_json,
          model_store_path = EXCLUDED.model_store_path,
          artifact_path = EXCLUDED.artifact_path,
          log_path = EXCLUDED.log_path,
          unified_schema = EXCLUDED.unified_schema,
          unified_table = EXCLUDED.unified_table,
          dataset_rows = EXCLUDED.dataset_rows,
          feature_cols = EXCLUDED.feature_cols,
          error_message = EXCLUDED.error_message,
          started_at = EXCLUDED.started_at,
          ended_at = EXCLUDED.ended_at;
        """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "config_id": payload.get("config_id"),
                "run_id": payload.get("run_id"),
                "status": payload.get("status"),
                "model_name": payload.get("model_name"),
                "horizon": payload.get("horizon"),
                "params_json": json.dumps(payload.get("params_json", {}), ensure_ascii=False),
                "exog_json": json.dumps(payload.get("exog_json", {}), ensure_ascii=False),
                "metrics_json": json.dumps(payload.get("metrics_json", {}), ensure_ascii=False),
                "diagnostics_json": json.dumps(payload.get("diagnostics_json", {}), ensure_ascii=False),
                "explain_json": json.dumps(payload.get("explain_json", {}), ensure_ascii=False),
                "model_save_json": json.dumps(payload.get("model_save_json", {}), ensure_ascii=False),
                "model_load_json": json.dumps(payload.get("model_load_json", {}), ensure_ascii=False),
                "model_analyze_json": json.dumps(payload.get("model_analyze_json", {}), ensure_ascii=False),
                "model_store_path": payload.get("model_store_path"),
                "artifact_path": payload.get("artifact_path"),
                "log_path": payload.get("log_path"),
                "unified_schema": payload.get("unified_schema"),
                "unified_table": payload.get("unified_table"),
                "dataset_rows": payload.get("dataset_rows"),
                "feature_cols": payload.get("feature_cols"),
                "error_message": payload.get("error_message"),
                "started_at": payload.get("started_at"),
                "ended_at": payload.get("ended_at"),
            },
        )


def _mark_meta_last(engine: Engine, config_id: int, status: str, run_id: str | None) -> None:
    schema = _safe_ident(settings.meta_schema)
    table = _safe_ident(settings.meta_table)
    q = text(
        f"""
        UPDATE {schema}.{table}
        SET last_status = :status,
            last_run_id = :run_id,
            last_run_at = now(),
            updated_at = now()
        WHERE config_id = :config_id
        """
    )
    with engine.begin() as conn:
        conn.execute(q, {"config_id": int(config_id), "status": status, "run_id": run_id})


def _unified_spec_from_row(row: dict[str, Any]) -> UnifiedBuildSpec:
    return UnifiedBuildSpec(
        profile="local",
        env="LOCAL",
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
        base_schema=str(row.get("base_schema") or "dataset"),
        base_table=str(row.get("base_table") or "loto_y_ts"),
        hist_schema=str(row.get("hist_schema") or "dataset"),
        hist_table=str(row.get("hist_table") or "loto_hist_feat"),
        exog_schema=str(row.get("exog_schema") or "exog"),
        output_schema=str(row.get("output_schema") or settings.exog_schema),
        output_table=str(row.get("output_table") or "loto_y_ts_unified"),
        output_if_exists=str(row.get("output_if_exists") or "replace"),
        output_csv_path=row.get("output_csv_path"),
        output_parquet_path=row.get("output_parquet_path"),
        output_spark_path=row.get("output_spark_path"),
        output_spark_format=str(row.get("output_spark_format") or "parquet"),
    )


def _resolve_save_path(row: dict[str, Any], run_id: str, model_name: str) -> Path:
    raw = row.get("save_path")
    if isinstance(raw, str) and raw.strip():
        try:
            path = raw.format(
                run_id=run_id,
                config_id=int(row.get("config_id") or 0),
                model_name=model_name,
            )
        except Exception:
            path = raw
        return Path(path).expanduser().resolve()
    return (settings.artifact_dir / "saved_models" / run_id).expanduser().resolve()


def _validate_unified_grouping(df, row: dict[str, Any], ds_spec: UnifiedBuildSpec) -> dict[str, Any]:
    group_cols = _normalize_group_cols(row.get("unified_group_cols_json"))
    present_cols = [c for c in group_cols if c in df.columns]
    missing_cols = [c for c in group_cols if c not in df.columns]
    time_col = str(ds_spec.time_col)
    time_present = time_col in df.columns

    null_key_rows = 0
    if present_cols:
        null_key_rows = int(df[present_cols].isna().any(axis=1).sum())

    dup_group_time_rows = 0
    if present_cols and time_present:
        dup_group_time_rows = int(df.duplicated(subset=[*present_cols, time_col], keep=False).sum())

    group_count = int(df[present_cols].drop_duplicates().shape[0]) if present_cols else 0
    ok = (not missing_cols) and null_key_rows == 0 and dup_group_time_rows == 0
    return {
        "ok": bool(ok),
        "group_cols": group_cols,
        "present_group_cols": present_cols,
        "missing_group_cols": missing_cols,
        "time_col": time_col,
        "time_col_present": bool(time_present),
        "row_count": int(len(df)),
        "group_count": int(group_count),
        "null_key_rows": int(null_key_rows),
        "duplicate_group_time_rows": int(dup_group_time_rows),
    }


def _apply_unified_filter(df, row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    filt = _to_json_dict(row.get("unified_filter_json"))
    if not filt:
        return df, {"enabled": False, "filters": {}, "rows_before": int(len(df)), "rows_after": int(len(df))}

    out = df
    applied: dict[str, Any] = {}
    missing_cols: list[str] = []
    rows_before = int(len(df))
    for key, val in filt.items():
        col = str(key)
        if col not in out.columns:
            missing_cols.append(col)
            continue
        applied[col] = val
        out = out.loc[out[col] == val]
    rows_after = int(len(out))
    info = {
        "enabled": True,
        "filters": applied,
        "missing_filter_cols": missing_cols,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }
    return out, info


def _is_postgres_column_limit_error(exc: Exception) -> bool:
    raw = str(exc).lower()
    return ("toomanycolumns" in raw) or ("tables can have at most 1600 columns" in raw)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_nf_meta_tables(engine: Engine) -> None:
    for sql_rel in [
        "sql/00_create_schema.sql",
        "sql/01_create_meta_tables.sql",
        "sql/02_create_catalog_and_grid_tables.sql",
        "sql/03_create_nf_automodel_tables.sql",
    ]:
        execute_sql_file(engine, str(_repo_root() / sql_rel))


def run_meta_automodel(
    config_id: int | None = None,
    limit: int = 100,
    stop_on_error: bool = False,
    ensure_db_init: bool = False,
    skip_existing_success: bool = True,
) -> dict[str, Any]:
    engine = make_engine()
    if ensure_db_init:
        _ensure_nf_meta_tables(engine)
    rows = _meta_rows(engine, config_id=config_id, limit=limit)
    if not rows:
        return {"executed": 0, "success": 0, "failed": 0, "message": "no active meta rows"}
    print(
        f"[meta-automodel-run] start active_configs={len(rows)} limit={int(limit)} stop_on_error={bool(stop_on_error)}",
        flush=True,
    )

    success = 0
    failed = 0
    skipped = 0
    completed_signatures = _load_success_combo_signatures(engine) if bool(skip_existing_success) else set()
    summaries: list[dict[str, Any]] = []

    for row in rows:
        cfg_id = int(row["config_id"])
        model_name = str(row.get("auto_cls_model") or row.get("model_name") or "AutoNHITS")
        auto_h = row.get("auto_h")
        horizon = _to_int(auto_h if auto_h is not None else row.get("horizon"), settings.default_horizon)
        run_predict = _to_bool(row.get("run_predict"), True)
        run_evaluate = _to_bool(row.get("run_evaluate"), True)
        run_explain = _to_bool(row.get("run_explain"), True)
        run_save = _to_bool(row.get("run_save"), True)
        run_load = _to_bool(row.get("run_load"), True)
        run_analyze = _to_bool(row.get("run_analyze"), True)
        save_dataset = _to_bool(row.get("save_dataset"), False)
        save_overwrite = _to_bool(row.get("save_overwrite"), True)
        load_check_predict = _to_bool(row.get("load_check_predict"), False)
        explain_repeats = _to_int(row.get("explain_repeats"), 3)
        max_tasks = row.get("max_tasks")
        recursive_depth = max(1, _to_int(row.get("recursive_depth"), 1))
        random_seed = _to_int(row.get("random_seed"), 1)
        strict_group_check = _to_bool(row.get("unified_group_validate_strict"), False)
        auto_backend = str(row.get("auto_backend") or "ray")
        auto_loss = str(row.get("auto_loss") or "MAE")
        auto_valid_loss = str(row.get("auto_valid_loss") or "MAE")
        if auto_valid_loss.strip().upper() != auto_loss.strip().upper():
            print(
                f"[meta-automodel-run] config_id={cfg_id} forcing auto_valid_loss={auto_loss} (was {auto_valid_loss})",
                flush=True,
            )
        auto_valid_loss = auto_loss
        auto_search_alg = str(row.get("auto_search_alg") or "BasicVariantGenerator")
        auto_num_samples = max(1, _to_int(row.get("auto_num_samples"), 10))
        auto_cpus = row.get("auto_cpus")
        auto_gpus = row.get("auto_gpus")
        auto_refit_with_val = _to_bool(row.get("auto_refit_with_val"), False)
        auto_verbose = _to_bool(row.get("auto_verbose"), False)
        auto_alias = row.get("auto_alias")
        auto_callbacks_json = _to_json_list(row.get("auto_callbacks_json"))
        auto_config_json = _to_json_dict(row.get("auto_config_json"))

        raw_fixed: dict[str, Any] = {}
        raw_fixed.update(auto_config_json)
        raw_fixed.update(_to_json_dict(row.get("model_params_json")))
        raw_param_space = _to_json_dict(row.get("param_space_json"))
        raw_param_mode = _to_json_dict(row.get("param_mode_json"))
        fixed_params, param_space, _ = _apply_param_mode(raw_fixed, raw_param_space, raw_param_mode)
        if "backend" not in fixed_params:
            fixed_params["backend"] = auto_backend
        if "num_samples" not in fixed_params:
            fixed_params["num_samples"] = auto_num_samples
        if "loss_name" not in fixed_params:
            fixed_params["loss_name"] = auto_loss
        forced_loss_name = str(fixed_params.get("loss_name") or auto_loss)
        if str(fixed_params.get("valid_loss_name") or "").strip().upper() != forced_loss_name.strip().upper():
            fixed_params["valid_loss_name"] = forced_loss_name
        if "search_alg_name" not in fixed_params:
            fixed_params["search_alg_name"] = auto_search_alg
        if auto_cpus is not None and "cpus" not in fixed_params:
            fixed_params["cpus"] = auto_cpus
        if auto_gpus is not None and "gpus" not in fixed_params:
            fixed_params["gpus"] = auto_gpus
        if "refit_with_val" not in fixed_params:
            fixed_params["refit_with_val"] = auto_refit_with_val
        if "verbose" not in fixed_params:
            fixed_params["verbose"] = auto_verbose
        if auto_alias is not None and "alias" not in fixed_params:
            fixed_params["alias"] = auto_alias
        if auto_callbacks_json and "callbacks" not in fixed_params:
            fixed_params["callbacks"] = auto_callbacks_json
        max_tasks_int = _to_int(max_tasks, 0) if max_tasks is not None else 0
        combos = expand_param_grid(param_space, max_tasks=max_tasks_int if max_tasks_int > 0 else None)
        if not combos:
            combos = [{}]
        total_tasks = int(len(combos) * recursive_depth)
        print(
            f"[meta-automodel-run] config_id={cfg_id} config_name={row.get('config_name')} "
            f"model={model_name} horizon={horizon} tasks={total_tasks}",
            flush=True,
        )
        arg_validation = _validate_meta_model_arguments(row)
        if not arg_validation.get("ok", False):
            validation_error = "; ".join([str(e) for e in arg_validation.get("errors", [])])
            raise ValueError(f"config_id={cfg_id} meta argument validation failed: {validation_error}")

        ds_spec = _unified_spec_from_row(row)
        print(
            f"[meta-automodel-run] config_id={cfg_id} building unified dataset "
            f"target={ds_spec.output_schema}.{ds_spec.output_table}",
            flush=True,
        )

        def _cfg_progress(message: str, _cfg_id: int = cfg_id) -> None:
            print(f"[meta-automodel-run] config_id={_cfg_id} {message}", flush=True)

        base_table_lower = str(ds_spec.base_table).lower()
        base_is_pre_unified = (
            "_unified" in base_table_lower or base_table_lower.endswith("_spark") or "_spark_" in base_table_lower
        )
        ds_build_spec = ds_spec
        if base_is_pre_unified:
            _cfg_progress(
                f"detected pre-unified base table={ds_spec.base_schema}.{ds_spec.base_table}; "
                "skip extra hist/exog joins for faster execution"
            )
            ds_build_spec = replace(
                ds_spec,
                hist_schema="__skip__",
                hist_table="__skip__",
                exog_schema="__skip__",
            )

        ds_result = build_unified_dataset(engine, ds_build_spec, progress=_cfg_progress)
        filtered_df, filter_info = _apply_unified_filter(ds_result.dataframe, row=row)
        _cfg_progress(
            "unified filter applied: "
            f"enabled={filter_info.get('enabled', False)} "
            f"rows_before={filter_info.get('rows_before', len(ds_result.dataframe))} "
            f"rows_after={filter_info.get('rows_after', len(filtered_df))} "
            f"cols={len(filtered_df.columns)}"
        )

        postgres_col_limit = 1600
        outputs: dict[str, Any]
        dataset_name = f"{ds_spec.output_schema}.{ds_spec.output_table}"
        if len(filtered_df.columns) > postgres_col_limit:
            reason = f"skip postgres persist: columns={len(filtered_df.columns)} > postgres_limit={postgres_col_limit}"
            _cfg_progress(reason)
            outputs = {
                "postgres": {
                    "schema": ds_spec.output_schema,
                    "table": ds_spec.output_table,
                    "rows": int(len(filtered_df)),
                    "columns": int(len(filtered_df.columns)),
                    "skipped": True,
                    "reason": reason,
                }
            }
            dataset_name = f"inmemory:{ds_spec.base_schema}.{ds_spec.base_table}"
        else:
            try:
                outputs = persist_unified_outputs(engine, filtered_df, ds_spec, progress=_cfg_progress)
            except Exception as e:
                if _is_postgres_column_limit_error(e):
                    reason = f"skip postgres persist after DB column-limit error: columns={len(filtered_df.columns)}"
                    _cfg_progress(reason)
                    outputs = {
                        "postgres": {
                            "schema": ds_spec.output_schema,
                            "table": ds_spec.output_table,
                            "rows": int(len(filtered_df)),
                            "columns": int(len(filtered_df.columns)),
                            "skipped": True,
                            "reason": reason,
                            "error": str(e),
                        }
                    }
                    dataset_name = f"inmemory:{ds_spec.base_schema}.{ds_spec.base_table}"
                else:
                    raise
        feature_cols = [c for c in filtered_df.columns if c.startswith(("hist_", "stat_", "feat_"))]
        group_check = _validate_unified_grouping(filtered_df, row=row, ds_spec=ds_spec)
        group_check_error: str | None = None
        filter_error: str | None = None
        if filter_info.get("enabled", False) and int(filter_info.get("rows_after", 0)) <= 0:
            filter_error = f"unified_filter_json produced 0 rows: {filter_info.get('filters', {})}"
        if strict_group_check and not group_check.get("ok", False):
            group_check_error = (
                "unified grouping validation failed: "
                f"missing={group_check.get('missing_group_cols')}, "
                f"null_key_rows={group_check.get('null_key_rows')}, "
                f"duplicate_group_time_rows={group_check.get('duplicate_group_time_rows')}"
            )

        cfg_runs = 0
        cfg_success = 0
        cfg_failed = 0
        cfg_skipped = 0

        for depth in range(recursive_depth):
            for idx, combo in enumerate(combos, start=1):
                cfg_runs += 1
                params = dict(fixed_params)
                params.update(combo)
                if "seed" not in params:
                    params["seed"] = int(random_seed + depth)
                combo_signature = _build_combo_signature(model_name=model_name, horizon=horizon, params_obj=params)
                if bool(skip_existing_success) and combo_signature in completed_signatures:
                    skipped += 1
                    cfg_skipped += 1
                    print(
                        f"[meta-automodel-run] skip already-success config_id={cfg_id} "
                        f"task={cfg_runs}/{total_tasks} signature={combo_signature[:12]}",
                        flush=True,
                    )
                    continue
                run_id = f"cfg{cfg_id}_d{depth + 1}_t{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                started_at = datetime.now(timezone.utc)
                print(
                    f"[meta-automodel-run] config_id={cfg_id} task={cfg_runs}/{total_tasks} "
                    f"run_id={run_id} seed={params.get('seed')}",
                    flush=True,
                )

                status = "success"
                err: str | None = None
                train_out: dict[str, Any] = {}
                eval_out: dict[str, Any] = {}
                explain_out: dict[str, Any] = {}
                model_save_out: dict[str, Any] = {}
                model_load_out: dict[str, Any] = {}
                model_analyze_out: dict[str, Any] = {}
                model_store_path: str | None = None

                try:
                    if filter_error:
                        raise ValueError(filter_error)
                    if group_check_error:
                        raise ValueError(group_check_error)
                    train_out = train(
                        model_name=model_name,
                        h=horizon,
                        model_params=params,
                        run_id=run_id,
                        dataset_df=filtered_df,
                        dataset_name=dataset_name,
                    )
                    if run_predict:
                        predict_with_dataset(train_out["run_id"], filtered_df, h=horizon)
                    if run_evaluate:
                        eval_out = evaluate(train_out["run_id"], dataset_df=filtered_df)
                    if run_explain:
                        imp = permutation_importance_exog(
                            train_out["run_id"],
                            h=horizon,
                            n_repeats=explain_repeats,
                            dataset_df=filtered_df,
                        )
                        explain_out = {
                            "feature_count": int(len(imp)),
                            "top_features": imp.head(30).to_dict(orient="records"),
                        }
                    if run_save or run_load or run_analyze:
                        from ..models.neuralforecast_model import save_load_analyze_model_bundle

                        store_path = _resolve_save_path(row, run_id=train_out["run_id"], model_name=model_name)
                        nf_save_kwargs = params.get("nf_save_kwargs")
                        nf_load_kwargs = params.get("nf_load_kwargs")
                        nf_predict_insample_kwargs = params.get("nf_predict_insample_kwargs")
                        model_ops = save_load_analyze_model_bundle(
                            run_id=train_out["run_id"],
                            source_dir=Path(train_out["artifact_path"]),
                            save_path=str(store_path),
                            run_save=run_save,
                            run_load=run_load,
                            run_analyze=run_analyze,
                            save_dataset=save_dataset,
                            save_overwrite=save_overwrite,
                            load_check_predict=load_check_predict,
                            insample_step_size=1,
                            save_kwargs=(dict(nf_save_kwargs) if isinstance(nf_save_kwargs, dict) else None),
                            load_kwargs=(dict(nf_load_kwargs) if isinstance(nf_load_kwargs, dict) else None),
                            predict_insample_kwargs=(
                                dict(nf_predict_insample_kwargs)
                                if isinstance(nf_predict_insample_kwargs, dict)
                                else None
                            ),
                        )
                        model_save_out = dict(model_ops.get("save", {}))
                        model_load_out = dict(model_ops.get("load", {}))
                        model_analyze_out = dict(model_ops.get("analyze", {}))
                        model_store_path = str(model_ops.get("store_path") or store_path)
                except Exception as e:
                    status = "failed"
                    err = str(e)

                ended_at = datetime.now(timezone.utc)
                run_elapsed = float((ended_at - started_at).total_seconds())
                _write_result(
                    engine,
                    {
                        "config_id": cfg_id,
                        "run_id": run_id,
                        "status": status,
                        "model_name": model_name,
                        "horizon": horizon,
                        "params_json": params,
                        "exog_json": train_out.get("exog", {}),
                        "metrics_json": (eval_out or {}).get("metrics", {}),
                        "diagnostics_json": {
                            "combo_signature": combo_signature,
                            **(
                                (eval_out or {}).get("diagnostics", {})
                                if isinstance((eval_out or {}).get("diagnostics", {}), dict)
                                else {}
                            ),
                            "unified_group_check": group_check,
                            "unified_filter": filter_info,
                        },
                        "explain_json": explain_out,
                        "model_save_json": model_save_out,
                        "model_load_json": model_load_out,
                        "model_analyze_json": model_analyze_out,
                        "model_store_path": model_store_path,
                        "artifact_path": train_out.get("artifact_path"),
                        "log_path": train_out.get("log_path"),
                        "unified_schema": ds_spec.output_schema,
                        "unified_table": ds_spec.output_table,
                        "dataset_rows": int(len(filtered_df)),
                        "feature_cols": int(len(feature_cols)),
                        "error_message": err,
                        "started_at": started_at,
                        "ended_at": ended_at,
                    },
                )
                _mark_meta_last(engine, cfg_id, status=status, run_id=run_id)

                if status == "success":
                    success += 1
                    cfg_success += 1
                    completed_signatures.add(combo_signature)
                    print(
                        f"[meta-automodel-run] success config_id={cfg_id} run_id={run_id} elapsed={run_elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    failed += 1
                    cfg_failed += 1
                    print(
                        f"[meta-automodel-run] failed config_id={cfg_id} run_id={run_id} "
                        f"elapsed={run_elapsed:.1f}s error={err}",
                        flush=True,
                    )
                    if stop_on_error:
                        return {
                            "executed": success + failed,
                            "success": success,
                            "failed": failed,
                            "skipped": skipped,
                            "stopped_on_error": True,
                            "skip_existing_success": bool(skip_existing_success),
                            "summaries": summaries,
                        }

        summaries.append(
            {
                "config_id": cfg_id,
                "config_name": row.get("config_name"),
                "runs": cfg_runs,
                "success": cfg_success,
                "failed": cfg_failed,
                "skipped": cfg_skipped,
                "dataset_rows": int(len(ds_result.dataframe)),
                "feature_cols": int(len(feature_cols)),
                "joined_tables": ds_result.joined_tables,
                "outputs": outputs,
            }
        )
        print(
            f"[meta-automodel-run] config done config_id={cfg_id} "
            f"success={cfg_success} failed={cfg_failed} skipped={cfg_skipped}",
            flush=True,
        )

    out = {
        "executed": success + failed,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "stopped_on_error": False,
        "ensure_db_init": bool(ensure_db_init),
        "skip_existing_success": bool(skip_existing_success),
        "summaries": summaries,
    }
    print(
        f"[meta-automodel-run] finished executed={out['executed']} "
        f"success={out['success']} failed={out['failed']} skipped={out['skipped']}",
        flush=True,
    )
    return out
