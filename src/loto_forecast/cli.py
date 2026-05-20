from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from .catalog.codegen_catalog import list_library_symbols, upsert_codegen_catalog, validate_call_arguments
from .config.settings import settings
from .data.db import execute_sql_file, make_engine, read_timeseries
from .infra.logging_utils import setup_logging
from .infra.meta_store import list_grid_tasks
from .models.registry import get_adapter, list_adapters
from .orchestration.grid_runner import create_grid, run_grid


def _load_json_any(raw: str | None, default: Any = None) -> Any:
    if raw is None:
        return default
    raw_text = str(raw).strip()
    if raw_text == "":
        return default
    try:
        parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(raw_text)
        return parsed
    except Exception:
        pass
    try:
        p = Path(raw_text)
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except OSError:
        # Long/invalid path-like input should be treated as malformed JSON input.
        pass
    raise ValueError("JSON argument must be JSON text, Python literal, or path to a JSON file")


def _load_json_arg(raw: str | None, default: dict | None = None) -> dict[str, Any]:
    parsed = _load_json_any(raw, default=dict(default or {}))
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must resolve to an object/dict")
    return dict(parsed)


def _resolve_run_artifact_path(source_path: str | None, run_id: str) -> Path:
    rid = str(run_id or "").strip()
    base = Path(source_path).expanduser() if source_path else (settings.artifact_dir / rid)
    candidates: list[Path] = [base]
    if rid and base.name != rid:
        candidates.insert(0, base / rid)
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    return candidates[0]


def _resolve_model_store_path(save_path: str | None, run_id: str) -> str | None:
    if save_path is None:
        return None
    rid = str(run_id or "").strip()
    raw = str(save_path).replace("{run_id}", rid)
    base = Path(raw).expanduser()
    candidates: list[Path] = [base]
    if rid and base.name != rid:
        candidates.insert(0, base / rid)
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return str(cand)
    return str(candidates[0])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _apply_neuralforecast_patches() -> None:
    from .patches.neuralforecast_autoformer_safe_topk import apply as apply_autoformer_safe_topk

    apply_autoformer_safe_topk()


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def _csv_values(raw: Any) -> list[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _apply_in_filter(df: Any, col: str, raw: Any):
    if col not in df.columns:
        return df
    vals = _csv_values(raw)
    if not vals:
        return df
    return df[df[col].astype(str).isin(vals)].copy()


def _infer_exog_from_table_for_model(model_name: str, params: dict[str, Any]) -> dict[str, list[str]]:
    from .features.engineering import infer_exog_columns
    from .models.neuralforecast_model import get_model_exog_support

    dataset_schema = str(params.get("dataset_schema", settings.db_schema) or settings.db_schema)
    dataset_table = str(params.get("dataset_table", settings.db_table) or settings.db_table)
    dataset_where = params.get("dataset_where")
    dataset_where = str(dataset_where).strip() if dataset_where is not None else None
    dataset_where = dataset_where if dataset_where else None
    target_loto = params.get("target_loto", "")
    target_unique_id = params.get("target_unique_id", "")
    target_ts_type = params.get("target_ts_type", "")

    engine = make_engine()
    raw = read_timeseries(engine, dataset_schema, dataset_table, where_sql=dataset_where)
    raw = _apply_in_filter(raw, "loto", target_loto)
    raw = _apply_in_filter(raw, settings.id_col, target_unique_id)
    raw = _apply_in_filter(raw, "ts_type", target_ts_type)
    inferred = infer_exog_columns(raw)

    support = get_model_exog_support(str(model_name))
    mapped = {
        "futr_exog_list": list(inferred.get("futr_exog", [])) if bool(support.get("futr", False)) else [],
        "hist_exog_list": list(inferred.get("hist_exog", [])) if bool(support.get("hist", False)) else [],
        "stat_exog_list": list(inferred.get("stat_exog", [])) if bool(support.get("stat", False)) else [],
    }
    logger.info(
        "auto exog inferred from table "
        f"{dataset_schema}.{dataset_table}: "
        f"futr={len(mapped['futr_exog_list'])} hist={len(mapped['hist_exog_list'])} stat={len(mapped['stat_exog_list'])}"
    )
    return mapped


def _db_init_sql_paths() -> list[Path]:
    return [
        _repo_root() / "sql" / "00_create_schema.sql",
        _repo_root() / "sql" / "01_create_meta_tables.sql",
        _repo_root() / "sql" / "02_create_catalog_and_grid_tables.sql",
        _repo_root() / "sql" / "03_create_nf_automodel_tables.sql",
        _repo_root() / "sql" / "04_create_log_tables.sql",
    ]


def cmd_db_init(args: argparse.Namespace) -> None:
    setup_logging("db_init")
    sql_paths = _db_init_sql_paths()
    if getattr(args, "dry_run", False):
        logger.info("DB init dry-run only. No SQL was executed.")
        for sql_path in sql_paths:
            logger.info(f"would apply: {sql_path}")
        return

    confirmed_by_flag = bool(getattr(args, "yes_i_understand_db_init_may_write", False))
    confirmed_by_env = os.getenv("LOTO_ALLOW_DB_INIT", "").strip() == "1"
    if not (confirmed_by_flag and confirmed_by_env):
        raise SystemExit(
            "db-init may create or alter DB objects. Refusing to run without both "
            "--yes-i-understand-db-init-may-write and LOTO_ALLOW_DB_INIT=1. "
            "Run with --dry-run first and confirm a backup before applying."
        )

    engine = make_engine()
    for sql_path in sql_paths:
        execute_sql_file(engine, str(sql_path))
        logger.info(f"applied: {sql_path}")
    logger.info("DB init done")


def cmd_train(args: argparse.Namespace) -> None:
    _apply_neuralforecast_patches()
    from .orchestration.pipeline import train

    params = _load_json_arg(args.params_json, default={})
    auto_exog_from_table = bool(params.pop("auto_exog_from_table", True))
    if getattr(args, "auto_exog_from_table", None) is not None:
        auto_exog_from_table = bool(args.auto_exog_from_table)
    dataset_input_method = str(params.get("dataset_input_method", "db_table") or "db_table").strip().lower()
    if auto_exog_from_table and dataset_input_method != "db_table":
        logger.info(f"auto exog inference skipped: dataset_input_method={dataset_input_method} (requires db_table)")
        auto_exog_from_table = False
    if args.search_alg_name is not None:
        params["search_alg_name"] = args.search_alg_name
    if args.cpus is not None:
        params["cpus"] = int(args.cpus)
    if args.gpus is not None:
        params["gpus"] = int(args.gpus)
    if args.refit_with_val is not None:
        params["refit_with_val"] = bool(args.refit_with_val)
    if args.verbose is not None:
        params["verbose"] = bool(args.verbose)
    if args.strict_exog is not None:
        params["strict_exog"] = bool(args.strict_exog)
    if args.run_cross_validation is not None:
        params["run_cross_validation"] = bool(args.run_cross_validation)
    cli_exog_inputs: dict[str, tuple[bool, list[str]]] = {}
    for arg_name, key_name in [
        ("futr_exog_list_json", "futr_exog_list"),
        ("hist_exog_list_json", "hist_exog_list"),
        ("stat_exog_list_json", "stat_exog_list"),
    ]:
        raw = getattr(args, arg_name)
        provided = raw is not None
        if not provided:
            cli_exog_inputs[key_name] = (False, [])
            continue
        parsed = _load_json_any(raw, default=[])
        if not isinstance(parsed, list):
            raise ValueError(f"{arg_name.replace('_', '-')} must resolve to JSON array/list")
        parsed_list = [str(x).strip() for x in parsed if str(x).strip()]
        cli_exog_inputs[key_name] = (True, parsed_list)

    for key_name in ("futr_exog_list", "hist_exog_list", "stat_exog_list"):
        provided, parsed_list = cli_exog_inputs.get(key_name, (False, []))
        if provided:
            if parsed_list:
                params[key_name] = parsed_list
            elif not auto_exog_from_table:
                params[key_name] = []

    if auto_exog_from_table:
        needs_auto = {
            key_name: len(_as_str_list(params.get(key_name))) == 0
            for key_name in ("futr_exog_list", "hist_exog_list", "stat_exog_list")
        }
        if any(needs_auto.values()):
            try:
                inferred = _infer_exog_from_table_for_model(args.model, params)
                for key_name, need in needs_auto.items():
                    if not need:
                        continue
                    inferred_list = list(inferred.get(key_name, []))
                    if inferred_list:
                        params[key_name] = inferred_list
            except Exception as e:
                logger.warning(f"auto exog inference failed; fallback to provided exog lists: {e}")
    if args.nf_fit_kwargs_json is not None:
        params["nf_fit_kwargs"] = _load_json_arg(args.nf_fit_kwargs_json, default={})
    if args.nf_predict_kwargs_json is not None:
        params["nf_predict_kwargs"] = _load_json_arg(args.nf_predict_kwargs_json, default={})
    if args.nf_cross_validation_kwargs_json is not None:
        params["nf_cross_validation_kwargs"] = _load_json_arg(args.nf_cross_validation_kwargs_json, default={})
    if args.nf_save_kwargs_json is not None:
        params["nf_save_kwargs"] = _load_json_arg(args.nf_save_kwargs_json, default={})
    if args.nf_load_kwargs_json is not None:
        params["nf_load_kwargs"] = _load_json_arg(args.nf_load_kwargs_json, default={})
    if args.nf_predict_insample_kwargs_json is not None:
        params["nf_predict_insample_kwargs"] = _load_json_arg(args.nf_predict_insample_kwargs_json, default={})
    try:
        out = train(args.model, args.h, model_params=params, run_id=args.run_id)
        artifact_path = Path(str(out.get("artifact_path", ""))).expanduser()
        meta_path = artifact_path / "meta.json"
        summary = {
            "status": "success",
            "run_id": out.get("run_id"),
            "model_name": out.get("model_name"),
            "artifact_path": str(artifact_path),
            "artifact_exists": bool(artifact_path.exists()),
            "meta_exists": bool(meta_path.exists()),
            "log_path": out.get("log_path"),
            "exog": out.get("exog", {}),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as e:
        fail = {
            "status": "failed",
            "model_name": str(args.model),
            "error": str(e),
        }
        print(json.dumps(fail, ensure_ascii=False, indent=2))
        raise


def cmd_retrain(args: argparse.Namespace) -> None:
    _apply_neuralforecast_patches()
    from .orchestration.pipeline import retrain

    params = _load_json_arg(args.params_json, default={}) if args.params_json else None
    out = retrain(args.base_run_id, h=args.h, model_params=params)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    from .orchestration.pipeline import predict

    out = predict(
        args.run_id,
        args.h,
        dataset_input_method=args.dataset_input_method,
        dataset_schema=args.dataset_schema,
        dataset_table=args.dataset_table,
        dataset_where=args.dataset_where,
        dataset_sql=args.dataset_sql,
        dataset_path=args.dataset_path,
        dataframe_backend=args.dataframe_backend,
    )
    print(out.head(50).to_string(index=False))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .orchestration.pipeline import evaluate

    print(
        json.dumps(
            evaluate(
                args.run_id,
                step_eval_size=args.step_eval_size,
                dataset_input_method=args.dataset_input_method,
                dataset_schema=args.dataset_schema,
                dataset_table=args.dataset_table,
                dataset_where=args.dataset_where,
                dataset_sql=args.dataset_sql,
                dataset_path=args.dataset_path,
                dataframe_backend=args.dataframe_backend,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_explain(args: argparse.Namespace) -> None:
    from .analysis.explain import exog_granger_screening, neuralforecast_explainability, permutation_importance_exog

    if args.method == "permutation":
        print(permutation_importance_exog(args.run_id).head(50).to_string(index=False))
    elif args.method == "granger":
        print(exog_granger_screening(maxlag=args.maxlag, top_k=args.top_k).to_string(index=False))
    else:
        print(neuralforecast_explainability(args.run_id))


def cmd_catalog_import(args: argparse.Namespace) -> None:
    setup_logging("catalog_import")
    engine = make_engine()
    yaml_path = args.yaml_path or settings.codegen_yaml_path
    out = upsert_codegen_catalog(
        engine,
        yaml_path=yaml_path,
        library_name=args.library,
        replace_library=not args.append,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_catalog_validate(args: argparse.Namespace) -> None:
    engine = make_engine()
    arguments = _load_json_arg(args.arguments_json, default={})
    out = validate_call_arguments(
        engine,
        library_name=args.library,
        full_path=args.full_path,
        arguments=arguments,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_catalog_list(args: argparse.Namespace) -> None:
    engine = make_engine()
    rows = list_library_symbols(engine, args.library, limit=args.limit)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def cmd_grid_create(args: argparse.Namespace) -> None:
    param_space = _load_json_arg(args.param_space_json)
    exog_policy = _load_json_arg(args.exog_policy_json, default={})
    out = create_grid(
        grid_id=args.grid_id,
        library_name=args.library,
        adapter_name=args.adapter,
        model_name=args.model,
        horizon=args.h,
        param_space=param_space,
        exog_policy=exog_policy,
        run_predict=not args.no_predict,
        run_evaluate=not args.no_evaluate,
        max_tasks=args.max_tasks,
        note=args.note,
        created_by=args.created_by,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_grid_run(args: argparse.Namespace) -> None:
    out = run_grid(grid_id=args.grid_id, stop_on_error=args.stop_on_error)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_grid_status(args: argparse.Namespace) -> None:
    engine = make_engine()
    rows = list_grid_tasks(engine, args.grid_id, status=args.status, limit=args.limit)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def cmd_adapters(_: argparse.Namespace) -> None:
    info: dict[str, Any] = {"adapters": list_adapters()}
    for name in list_adapters():
        ad = get_adapter(name)
        info[name] = {
            "library": ad.library_name,
            "models": ad.list_models(),
        }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def cmd_build_exog(args: argparse.Namespace) -> None:
    from resources.exog_pipeline import ExogBuildSpec, run_exog_build

    group_cols = tuple([x.strip() for x in args.group_cols.split(",") if x.strip()])
    if not group_cols:
        raise ValueError("group-cols must not be empty")
    pyod_detectors = tuple([x.strip() for x in str(args.pyod_detectors).split(",") if x.strip()])
    merlion_models = tuple([x.strip() for x in str(args.merlion_models).split(",") if x.strip()])
    pypots_models = tuple([x.strip() for x in str(args.pypots_models).split(",") if x.strip()])
    tsfel_domains = tuple([x.strip() for x in str(args.tsfel_domains).split(",") if x.strip()])
    autogluon_generators = tuple([x.strip() for x in str(args.autogluon_generators).split(",") if x.strip()])

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
        group_cols=group_cols,
        time_col=args.time_col,
        target_col=args.target_col,
        parallel_workers=args.parallel_workers,
        enable_gpu_compute=args.enable_gpu_compute,
        enable_anomaly_features=args.enable_anomaly_features,
        pyod_codegen_yaml=args.pyod_codegen_yaml,
        pyod_detectors=pyod_detectors,
        pyod_contamination=args.pyod_contamination,
        anomaly_min_train_size=args.anomaly_min_train_size,
        anomaly_rolling_window=args.anomaly_rolling_window,
        enable_merlion_features=args.enable_merlion_features,
        merlion_codegen_yaml=args.merlion_codegen_yaml,
        merlion_models=merlion_models,
        merlion_contamination=args.merlion_contamination,
        merlion_min_train_size=args.merlion_min_train_size,
        merlion_n_estimators=args.merlion_n_estimators,
        merlion_max_n_samples=args.merlion_max_n_samples,
        merlion_random_state=args.merlion_random_state,
        enable_pypots_features=args.enable_pypots_features,
        pypots_codegen_yaml=args.pypots_codegen_yaml,
        pypots_models=pypots_models,
        pypots_anomaly_rate=args.pypots_anomaly_rate,
        pypots_window_size=args.pypots_window_size,
        pypots_min_train_windows=args.pypots_min_train_windows,
        pypots_epochs=args.pypots_epochs,
        pypots_batch_size=args.pypots_batch_size,
        enable_tsfel_features=args.enable_tsfel_features,
        tsfel_codegen_yaml=args.tsfel_codegen_yaml,
        tsfel_domains=tsfel_domains,
        tsfel_max_features=args.tsfel_max_features,
        tsfel_window_size=args.tsfel_window_size,
        tsfel_min_train_windows=args.tsfel_min_train_windows,
        tsfel_fill_method=args.tsfel_fill_method,
        tsfel_sampling_frequency=args.tsfel_sampling_frequency,
        enable_autogluon_features=args.enable_autogluon_features,
        autogluon_codegen_yaml=args.autogluon_codegen_yaml,
        autogluon_generators=autogluon_generators,
        autogluon_window_size=args.autogluon_window_size,
        autogluon_min_train_windows=args.autogluon_min_train_windows,
        autogluon_fill_method=args.autogluon_fill_method,
        autogluon_max_features=args.autogluon_max_features,
        enable_stumpy_features=args.enable_stumpy_features,
        stumpy_codegen_yaml=args.stumpy_codegen_yaml,
        stumpy_window_size=args.stumpy_window_size,
        stumpy_min_train_windows=args.stumpy_min_train_windows,
        stumpy_fill_method=args.stumpy_fill_method,
        stumpy_discord_quantile=args.stumpy_discord_quantile,
        enable_tsfresh_features=args.enable_tsfresh_features,
        tsfresh_codegen_yaml=args.tsfresh_codegen_yaml,
        tsfresh_feature_set=args.tsfresh_feature_set,
        tsfresh_window_size=args.tsfresh_window_size,
        tsfresh_min_train_windows=args.tsfresh_min_train_windows,
        tsfresh_fill_method=args.tsfresh_fill_method,
        tsfresh_max_features=args.tsfresh_max_features,
        tsfresh_n_jobs=args.tsfresh_n_jobs,
        sampling_interval_sec=args.sampling_interval_sec,
        lib_docs_dir=args.lib_docs_dir,
    )
    out = run_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_exog_uni2ts(args: argparse.Namespace) -> None:
    from resources.uni2ts_exog_pipeline import Uni2TSExogSpec, run_uni2ts_exog_build

    group_cols = tuple([x.strip() for x in args.group_cols.split(",") if x.strip()])
    if not group_cols:
        raise ValueError("group-cols must not be empty")

    spec = Uni2TSExogSpec(
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
        group_cols=group_cols,
        time_col=args.time_col,
        target_col=args.target_col,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        parallel_workers=args.parallel_workers,
        model_name=args.model_name,
        model_version=args.model_version,
        model_checkpoint=args.model_checkpoint,
        local_files_only=args.local_files_only,
        enable_gpu_compute=args.enable_gpu_compute,
        sampling_interval_sec=args.sampling_interval_sec,
        uni2ts_codegen_yaml=args.uni2ts_codegen_yaml,
    )
    out = run_uni2ts_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_exog_timesfm(args: argparse.Namespace) -> None:
    from resources.timesfm_exog_pipeline import TimesFMExogSpec, run_timesfm_exog_build

    group_cols = tuple([x.strip() for x in args.group_cols.split(",") if x.strip()])
    if not group_cols:
        raise ValueError("group-cols must not be empty")

    loto_filter = tuple([x.strip() for x in (args.loto_filter or "").split(",") if x.strip()])
    ts_type_filter = tuple([x.strip() for x in (args.ts_type_filter or "").split(",") if x.strip()])

    spec = TimesFMExogSpec(
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
        only_missing=args.only_missing,
        group_cols=group_cols,
        time_col=args.time_col,
        target_col=args.target_col,
        source_row_id_column=args.source_row_id_column,
        y_idx_order_column=args.y_idx_order_column,
        ds_start=args.ds_start,
        ds_end=args.ds_end,
        loto_filter=loto_filter,
        ts_type_filter=ts_type_filter,
        backend=args.backend,
        model_id=args.model_id,
        model_name=args.model_name,
        model_version=args.model_version,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        min_points=args.min_points,
        batch_size=args.batch_size,
        normalize_method=args.normalize_method,
        fill_method=args.fill_method,
        parallel_workers=args.parallel_workers,
        enable_gpu_compute=args.enable_gpu_compute,
        local_files_only=args.local_files_only,
        sampling_interval_sec=args.sampling_interval_sec,
        postgres_copy_strategy=args.postgres_copy_strategy,
        timesfm_codegen_yaml=args.timesfm_codegen_yaml,
    )
    out = run_timesfm_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_exog_chronos(args: argparse.Namespace) -> None:
    from resources.chronos_exog_pipeline import ChronosExogSpec, run_chronos_exog_build

    group_cols = tuple([x.strip() for x in args.group_cols.split(",") if x.strip()])
    if not group_cols:
        raise ValueError("group-cols must not be empty")

    loto_filter = tuple([x.strip() for x in (args.loto_filter or "").split(",") if x.strip()])
    ts_type_filter = tuple([x.strip() for x in (args.ts_type_filter or "").split(",") if x.strip()])

    spec = ChronosExogSpec(
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
        only_missing=args.only_missing,
        group_cols=group_cols,
        time_col=args.time_col,
        target_col=args.target_col,
        source_row_id_column=args.source_row_id_column,
        y_idx_order_column=args.y_idx_order_column,
        ds_start=args.ds_start,
        ds_end=args.ds_end,
        loto_filter=loto_filter,
        ts_type_filter=ts_type_filter,
        backend=args.backend,
        model_id=args.model_id,
        model_name=args.model_name,
        model_version=args.model_version,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        min_points=args.min_points,
        batch_size=args.batch_size,
        normalize_method=args.normalize_method,
        fill_method=args.fill_method,
        parallel_workers=args.parallel_workers,
        enable_gpu_compute=args.enable_gpu_compute,
        local_files_only=args.local_files_only,
        sampling_interval_sec=args.sampling_interval_sec,
        chronos_codegen_yaml=args.chronos_codegen_yaml,
    )
    out = run_chronos_exog_build(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_build_unified_dataset(args: argparse.Namespace) -> None:
    from .data.unified_dataset import UnifiedBuildSpec, run_unified_dataset_build

    include_exog_tables = tuple([x.strip() for x in (args.include_exog_tables or "").split(",") if x.strip()])
    exclude_exog_tables = tuple([x.strip() for x in (args.exclude_exog_tables or "").split(",") if x.strip()])
    key_candidates = tuple([x.strip() for x in (args.key_candidates or "").split(",") if x.strip()])
    if not key_candidates:
        key_candidates = ("loto_y_ts_row_id", "row_id", "unique_id", "loto", "ts_type", "ds")

    spec = UnifiedBuildSpec(
        profile=args.profile,
        env=args.env,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        base_schema=args.base_schema,
        base_table=args.base_table,
        hist_schema=args.hist_schema,
        hist_table=args.hist_table,
        exog_schema=args.exog_schema,
        id_col=args.id_col,
        time_col=args.time_col,
        target_col=args.target_col,
        key_candidates=key_candidates,
        include_exog_tables=include_exog_tables,
        exclude_exog_tables=exclude_exog_tables,
        output_schema=args.output_schema,
        output_table=args.output_table,
        output_if_exists=args.output_if_exists,
        output_csv_path=args.output_csv_path,
        output_parquet_path=args.output_parquet_path,
        output_spark_path=args.output_spark_path,
        output_spark_format=args.output_spark_format,
        sort_output=args.sort_output,
        create_postgres_index=args.create_postgres_index,
        postgres_chunksize=args.postgres_chunksize,
        postgres_write_mode=args.postgres_write_mode,
        postgres_copy_chunk_rows=args.postgres_copy_chunk_rows,
        fast_mode=args.fast_mode,
    )
    out = run_unified_dataset_build(spec, show_progress=args.show_progress)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_check_unified_grouping(args: argparse.Namespace) -> None:
    from .data.unified_dataset import check_unified_grouping_in_table

    group_cols = tuple([x.strip() for x in str(args.group_cols).split(",") if x.strip()])
    if not group_cols:
        raise ValueError("group-cols must not be empty")
    try:
        out = check_unified_grouping_in_table(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
            schema=args.schema,
            table=args.table,
            group_cols=group_cols,
            time_col=args.time_col,
            sample_limit=args.sample_limit,
        )
    except Exception as e:
        out = {
            "ok": False,
            "schema": args.schema,
            "table": args.table,
            "group_cols": list(group_cols),
            "time_col": args.time_col,
            "error": str(e),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_meta_automodel_create(args: argparse.Namespace) -> None:
    from .orchestration.meta_automodel import create_meta_automodel_config

    if bool(args.ensure_db_init):
        cmd_db_init(argparse.Namespace())

    config = _load_json_arg(args.config_json, default={}) if args.config_json else {}
    overrides: dict[str, Any] = {}

    if args.config_name is not None:
        overrides["config_name"] = args.config_name
    if args.active is not None:
        overrides["active"] = args.active
    if args.priority is not None:
        overrides["priority"] = args.priority
    if args.base_schema is not None:
        overrides["base_schema"] = args.base_schema
    if args.base_table is not None:
        overrides["base_table"] = args.base_table
    if args.hist_schema is not None:
        overrides["hist_schema"] = args.hist_schema
    if args.hist_table is not None:
        overrides["hist_table"] = args.hist_table
    if args.exog_schema is not None:
        overrides["exog_schema"] = args.exog_schema
    if args.output_schema is not None:
        overrides["output_schema"] = args.output_schema
    if args.output_table is not None:
        overrides["output_table"] = args.output_table
    if args.output_if_exists is not None:
        overrides["output_if_exists"] = args.output_if_exists
    if args.output_csv_path is not None:
        overrides["output_csv_path"] = args.output_csv_path
    if args.output_parquet_path is not None:
        overrides["output_parquet_path"] = args.output_parquet_path
    if args.output_spark_path is not None:
        overrides["output_spark_path"] = args.output_spark_path
    if args.output_spark_format is not None:
        overrides["output_spark_format"] = args.output_spark_format
    if args.unified_filter_json is not None:
        overrides["unified_filter_json"] = _load_json_arg(args.unified_filter_json, default={})
    if args.unified_group_cols_json is not None:
        parsed_group_cols = _load_json_any(args.unified_group_cols_json, default=["loto", "unique_id", "ts_type"])
        if not isinstance(parsed_group_cols, list):
            raise ValueError("unified_group_cols_json must resolve to a JSON array/list")
        overrides["unified_group_cols_json"] = parsed_group_cols
    if args.unified_group_validate_strict is not None:
        overrides["unified_group_validate_strict"] = args.unified_group_validate_strict
    if args.model_name is not None:
        overrides["model_name"] = args.model_name
    if args.h is not None:
        overrides["horizon"] = args.h
    if args.auto_cls_model is not None:
        overrides["auto_cls_model"] = args.auto_cls_model
    if args.auto_h is not None:
        overrides["auto_h"] = args.auto_h
    if args.auto_loss is not None:
        overrides["auto_loss"] = args.auto_loss
    if args.auto_valid_loss is not None:
        overrides["auto_valid_loss"] = args.auto_valid_loss
    if args.auto_config_json is not None:
        overrides["auto_config_json"] = _load_json_arg(args.auto_config_json, default={})
    if args.auto_search_alg is not None:
        overrides["auto_search_alg"] = args.auto_search_alg
    if args.auto_num_samples is not None:
        overrides["auto_num_samples"] = args.auto_num_samples
    if args.auto_cpus is not None:
        overrides["auto_cpus"] = args.auto_cpus
    if args.auto_gpus is not None:
        overrides["auto_gpus"] = args.auto_gpus
    if args.auto_refit_with_val is not None:
        overrides["auto_refit_with_val"] = args.auto_refit_with_val
    if args.auto_verbose is not None:
        overrides["auto_verbose"] = args.auto_verbose
    if args.auto_alias is not None:
        overrides["auto_alias"] = args.auto_alias
    if args.auto_backend is not None:
        overrides["auto_backend"] = args.auto_backend
    if args.auto_callbacks_json is not None:
        parsed_callbacks = _load_json_any(args.auto_callbacks_json, default=[])
        if not isinstance(parsed_callbacks, list):
            raise ValueError("auto_callbacks_json must resolve to a JSON array/list")
        overrides["auto_callbacks_json"] = parsed_callbacks
    model_params_overrides: dict[str, Any] = {}
    if args.model_params_json is not None:
        model_params_overrides.update(_load_json_arg(args.model_params_json, default={}))
    if args.strict_exog is not None:
        model_params_overrides["strict_exog"] = bool(args.strict_exog)
    if args.run_cross_validation is not None:
        model_params_overrides["run_cross_validation"] = bool(args.run_cross_validation)
    if args.futr_exog_list_json is not None:
        parsed_futr = _load_json_any(args.futr_exog_list_json, default=[])
        if not isinstance(parsed_futr, list):
            raise ValueError("futr-exog-list-json must resolve to JSON array/list")
        model_params_overrides["futr_exog_list"] = parsed_futr
    if args.hist_exog_list_json is not None:
        parsed_hist = _load_json_any(args.hist_exog_list_json, default=[])
        if not isinstance(parsed_hist, list):
            raise ValueError("hist-exog-list-json must resolve to JSON array/list")
        model_params_overrides["hist_exog_list"] = parsed_hist
    if args.stat_exog_list_json is not None:
        parsed_stat = _load_json_any(args.stat_exog_list_json, default=[])
        if not isinstance(parsed_stat, list):
            raise ValueError("stat-exog-list-json must resolve to JSON array/list")
        model_params_overrides["stat_exog_list"] = parsed_stat
    if args.nf_fit_kwargs_json is not None:
        model_params_overrides["nf_fit_kwargs"] = _load_json_arg(args.nf_fit_kwargs_json, default={})
    if args.nf_predict_kwargs_json is not None:
        model_params_overrides["nf_predict_kwargs"] = _load_json_arg(args.nf_predict_kwargs_json, default={})
    if args.nf_cross_validation_kwargs_json is not None:
        model_params_overrides["nf_cross_validation_kwargs"] = _load_json_arg(
            args.nf_cross_validation_kwargs_json, default={}
        )
    if args.nf_save_kwargs_json is not None:
        model_params_overrides["nf_save_kwargs"] = _load_json_arg(args.nf_save_kwargs_json, default={})
    if args.nf_load_kwargs_json is not None:
        model_params_overrides["nf_load_kwargs"] = _load_json_arg(args.nf_load_kwargs_json, default={})
    if args.nf_predict_insample_kwargs_json is not None:
        model_params_overrides["nf_predict_insample_kwargs"] = _load_json_arg(
            args.nf_predict_insample_kwargs_json,
            default={},
        )
    if model_params_overrides:
        base_model_params = config.get("model_params_json", {})
        if not isinstance(base_model_params, dict):
            base_model_params = {}
        merged_model_params = dict(base_model_params)
        merged_model_params.update(model_params_overrides)
        overrides["model_params_json"] = merged_model_params
    if args.param_space_json is not None:
        overrides["param_space_json"] = _load_json_arg(args.param_space_json, default={})
    if args.param_mode_json is not None:
        overrides["param_mode_json"] = _load_json_arg(args.param_mode_json, default={})
    if args.random_seed is not None:
        overrides["random_seed"] = args.random_seed
    if args.max_tasks is not None:
        overrides["max_tasks"] = args.max_tasks
    if args.recursive_depth is not None:
        overrides["recursive_depth"] = args.recursive_depth
    if args.run_predict is not None:
        overrides["run_predict"] = args.run_predict
    if args.run_evaluate is not None:
        overrides["run_evaluate"] = args.run_evaluate
    if args.run_explain is not None:
        overrides["run_explain"] = args.run_explain
    if args.run_save is not None:
        overrides["run_save"] = args.run_save
    if args.run_load is not None:
        overrides["run_load"] = args.run_load
    if args.run_analyze is not None:
        overrides["run_analyze"] = args.run_analyze
    if args.explain_repeats is not None:
        overrides["explain_repeats"] = args.explain_repeats
    if args.save_dataset is not None:
        overrides["save_dataset"] = args.save_dataset
    if args.save_overwrite is not None:
        overrides["save_overwrite"] = args.save_overwrite
    if args.save_path is not None:
        overrides["save_path"] = args.save_path
    if args.load_check_predict is not None:
        overrides["load_check_predict"] = args.load_check_predict
    if args.note is not None:
        overrides["note"] = args.note

    config.update(overrides)
    out = create_meta_automodel_config(config=config, upsert_by_name=not args.no_upsert_by_name)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_meta_automodel_run(args: argparse.Namespace) -> None:
    from .orchestration.meta_automodel import run_meta_automodel

    out = run_meta_automodel(
        config_id=args.config_id,
        limit=args.limit,
        stop_on_error=args.stop_on_error,
        ensure_db_init=args.ensure_db_init,
        skip_existing_success=args.skip_existing_success,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    failed = int(out.get("failed", 0) or 0)
    if failed > 0 and not bool(args.allow_failures):
        raise SystemExit(2)


def cmd_meta_automodel_arg_spec(args: argparse.Namespace) -> None:
    from .models.registry import get_adapter

    adapter = get_adapter("neuralforecast_auto")
    out = adapter.validate(model_name=args.model_name, model_params={})
    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "ok": bool(out.get("ok", False)),
                "accepted_params": out.get("accepted_params", []),
                "required_model_params": out.get("required_model_params", []),
                "reserved_param_specs": out.get("reserved_param_specs", {}),
                "model_exog_support": out.get("model_exog_support", {}),
                "model_exog_support_table": out.get("model_exog_support_table", []),
                "errors": out.get("errors", []),
                "warnings": out.get("warnings", []),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def cmd_meta_automodel_report(args: argparse.Namespace) -> None:
    from .analysis.meta_automodel_report import generate_meta_automodel_report

    out = generate_meta_automodel_report(
        config_id=args.config_id,
        run_id=args.run_id,
        status=args.status,
        limit=args.limit,
        target_metric=args.target_metric,
        higher_is_better=args.higher_is_better,
        recursive_depth=args.recursive_depth,
        min_group_size=args.min_group_size,
        alpha=args.alpha,
        out_dir=args.out_dir,
        top_k=args.top_k,
        write_outputs=args.write_outputs,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    if args.fail_on_empty and int(out.get("rows", 0)) <= 0:
        raise SystemExit(2)


def cmd_run_table_pyspark(args: argparse.Namespace) -> None:
    from .data.spark_table_runner import SparkTableRunSpec, run_table_with_pyspark

    spec = SparkTableRunSpec(
        profile=args.profile,
        env=args.env,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        source_schema=args.source_schema,
        source_table=args.source_table,
        source_sql=args.source_sql,
        source_view=args.source_view,
        transform_sql=args.transform_sql,
        repartition=args.repartition,
        target_schema=args.target_schema,
        target_table=args.target_table,
        output_if_exists=args.output_if_exists,
        output_parquet_path=args.output_parquet_path,
        output_csv_path=args.output_csv_path,
        output_spark_path=args.output_spark_path,
        output_spark_format=args.output_spark_format,
        spark_master=args.spark_master,
        app_name=args.app_name,
        execution_backend=args.execution_backend,
        dask_npartitions=args.dask_npartitions,
        prefer_pandas=args.prefer_pandas,
        skip_row_count=args.skip_row_count,
        spark_ui_enabled=args.spark_ui_enabled,
        spark_shuffle_partitions=args.spark_shuffle_partitions,
        spark_reader_fetchsize=args.spark_reader_fetchsize,
        postgres_write_mode=args.postgres_write_mode,
        postgres_copy_chunk_rows=args.postgres_copy_chunk_rows,
        postgres_lock_timeout_ms=args.postgres_lock_timeout_ms,
    )
    out = run_table_with_pyspark(spec)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def cmd_model_save_load_analyze(args: argparse.Namespace) -> None:
    from .models.neuralforecast_model import save_load_analyze_model_bundle

    source_dir = _resolve_run_artifact_path(args.source_path, args.run_id)
    save_path = _resolve_model_store_path(args.save_path, args.run_id)
    save_kwargs = _load_json_arg(args.save_kwargs_json, default={}) if args.save_kwargs_json else None
    load_kwargs = _load_json_arg(args.load_kwargs_json, default={}) if args.load_kwargs_json else None
    predict_insample_kwargs = (
        _load_json_arg(args.predict_insample_kwargs_json, default={}) if args.predict_insample_kwargs_json else None
    )
    out = save_load_analyze_model_bundle(
        run_id=args.run_id,
        source_dir=source_dir,
        save_path=save_path,
        run_save=args.run_save,
        run_load=args.run_load,
        run_analyze=args.run_analyze,
        save_dataset=args.save_dataset,
        save_overwrite=args.save_overwrite,
        load_check_predict=args.load_check_predict,
        insample_step_size=args.insample_step_size,
        save_kwargs=save_kwargs,
        load_kwargs=load_kwargs,
        predict_insample_kwargs=predict_insample_kwargs,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loto_forecast")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_db = sp.add_parser("db-init", help="Create schema and all meta/catalog/grid tables")
    p_db.add_argument("--dry-run", action="store_true", help="List SQL files without executing them")
    p_db.add_argument(
        "--yes-i-understand-db-init-may-write",
        action="store_true",
        help="Required together with LOTO_ALLOW_DB_INIT=1 before executing SQL",
    )
    p_db.set_defaults(func=cmd_db_init)

    p_tr = sp.add_parser("train", help="Train AutoModel with exogenous variables")
    p_tr.add_argument("--model", default="AutoNHITS")
    p_tr.add_argument("--h", type=int, default=settings.default_horizon)
    p_tr.add_argument("--run-id", default=None)
    p_tr.add_argument("--params-json", default=None, help="JSON string or JSON file path for model params")
    p_tr.add_argument("--search-alg-name", default=None)
    p_tr.add_argument("--cpus", type=int, default=None)
    p_tr.add_argument("--gpus", type=int, default=None)
    p_tr.add_argument("--refit-with-val", action=argparse.BooleanOptionalAction, default=None)
    p_tr.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    p_tr.add_argument("--strict-exog", action=argparse.BooleanOptionalAction, default=None)
    p_tr.add_argument("--run-cross-validation", action=argparse.BooleanOptionalAction, default=None)
    p_tr.add_argument(
        "--auto-exog-from-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When exog lists are omitted or empty, infer futr/hist/stat exog from dataset table columns.",
    )
    p_tr.add_argument("--futr-exog-list-json", default=None, help="JSON string/file -> list[str]")
    p_tr.add_argument("--hist-exog-list-json", default=None, help="JSON string/file -> list[str]")
    p_tr.add_argument("--stat-exog-list-json", default=None, help="JSON string/file -> list[str]")
    p_tr.add_argument("--nf-fit-kwargs-json", default=None, help="JSON object for NeuralForecast.fit kwargs")
    p_tr.add_argument("--nf-predict-kwargs-json", default=None, help="JSON object for NeuralForecast.predict kwargs")
    p_tr.add_argument(
        "--nf-cross-validation-kwargs-json",
        default=None,
        help="JSON object for NeuralForecast.cross_validation kwargs",
    )
    p_tr.add_argument("--nf-save-kwargs-json", default=None, help="JSON object for NeuralForecast.save kwargs")
    p_tr.add_argument("--nf-load-kwargs-json", default=None, help="JSON object for NeuralForecast.load kwargs")
    p_tr.add_argument(
        "--nf-predict-insample-kwargs-json",
        default=None,
        help="JSON object for NeuralForecast.predict_insample kwargs",
    )
    p_tr.set_defaults(func=cmd_train)

    p_rt = sp.add_parser("retrain", help="Retrain from existing run meta")
    p_rt.add_argument("--base-run-id", required=True)
    p_rt.add_argument("--h", type=int, default=None)
    p_rt.add_argument("--params-json", default=None)
    p_rt.set_defaults(func=cmd_retrain)

    p_pr = sp.add_parser("predict", help="Predict future h steps using saved model")
    p_pr.add_argument("--run-id", required=True)
    p_pr.add_argument("--h", type=int, default=None)
    p_pr.add_argument(
        "--dataset-input-method", default="db_table", choices=["db_table", "db_sql", "csv", "parquet", "json"]
    )
    p_pr.add_argument("--dataset-schema", default=settings.db_schema)
    p_pr.add_argument("--dataset-table", default=settings.db_table)
    p_pr.add_argument("--dataset-where", default=None)
    p_pr.add_argument("--dataset-sql", default=None)
    p_pr.add_argument("--dataset-path", default=None)
    p_pr.add_argument("--dataframe-backend", default="pandas", choices=["pandas", "polars", "dask", "spark", "ray"])
    p_pr.set_defaults(func=cmd_predict)

    p_ev = sp.add_parser("evaluate", help="Evaluate holdout and diagnostics")
    p_ev.add_argument("--run-id", required=True)
    p_ev.add_argument(
        "--step-eval-size", type=int, default=1, help="Group horizon steps by this size when reporting step metrics"
    )
    p_ev.add_argument(
        "--dataset-input-method", default="db_table", choices=["db_table", "db_sql", "csv", "parquet", "json"]
    )
    p_ev.add_argument("--dataset-schema", default=settings.db_schema)
    p_ev.add_argument("--dataset-table", default=settings.db_table)
    p_ev.add_argument("--dataset-where", default=None)
    p_ev.add_argument("--dataset-sql", default=None)
    p_ev.add_argument("--dataset-path", default=None)
    p_ev.add_argument("--dataframe-backend", default="pandas", choices=["pandas", "polars", "dask", "spark", "ray"])
    p_ev.set_defaults(func=cmd_evaluate)

    p_ex = sp.add_parser("explain", help="Explain exogenous contributions")
    p_ex.add_argument("--run-id", required=True)
    p_ex.add_argument("--method", default="permutation", choices=["permutation", "neuralforecast", "granger"])
    p_ex.add_argument("--maxlag", type=int, default=8)
    p_ex.add_argument("--top-k", type=int, default=20)
    p_ex.set_defaults(func=cmd_explain)

    p_ci = sp.add_parser("catalog-import", help="Import <library>_all_codegen.yaml into DB catalog")
    p_ci.add_argument("--yaml-path", default=None)
    p_ci.add_argument("--library", default="neuralforecast")
    p_ci.add_argument("--append", action="store_true", help="append mode (no replacement)")
    p_ci.set_defaults(func=cmd_catalog_import)

    p_cv = sp.add_parser("catalog-validate", help="Validate call args against imported catalog signature")
    p_cv.add_argument("--library", required=True)
    p_cv.add_argument("--full-path", required=True)
    p_cv.add_argument("--arguments-json", required=True)
    p_cv.set_defaults(func=cmd_catalog_validate)

    p_cl = sp.add_parser("catalog-list", help="List imported symbols")
    p_cl.add_argument("--library", required=True)
    p_cl.add_argument("--limit", type=int, default=100)
    p_cl.set_defaults(func=cmd_catalog_list)

    p_gc = sp.add_parser("grid-create", help="Create grid-search definition and expanded task table")
    p_gc.add_argument("--grid-id", required=True)
    p_gc.add_argument("--library", default="neuralforecast")
    p_gc.add_argument("--adapter", default="neuralforecast_auto")
    p_gc.add_argument("--model", default="AutoNHITS")
    p_gc.add_argument("--h", type=int, default=settings.default_horizon)
    p_gc.add_argument(
        "--param-space-json", required=True, help='JSON string/file. ex: {"num_samples":[10,20],"seed":[1,2]}'
    )
    p_gc.add_argument("--exog-policy-json", default="{}")
    p_gc.add_argument("--max-tasks", type=int, default=None)
    p_gc.add_argument("--note", default=None)
    p_gc.add_argument("--created-by", default="local")
    p_gc.add_argument("--no-predict", action="store_true")
    p_gc.add_argument("--no-evaluate", action="store_true")
    p_gc.set_defaults(func=cmd_grid_create)

    p_gr = sp.add_parser("grid-run", help="Run pending tasks for a grid")
    p_gr.add_argument("--grid-id", required=True)
    p_gr.add_argument("--stop-on-error", action="store_true")
    p_gr.set_defaults(func=cmd_grid_run)

    p_gs = sp.add_parser("grid-status", help="Show task status for a grid")
    p_gs.add_argument("--grid-id", required=True)
    p_gs.add_argument("--status", default=None)
    p_gs.add_argument("--limit", type=int, default=100)
    p_gs.set_defaults(func=cmd_grid_status)

    p_ad = sp.add_parser("adapters", help="List adapters and supported models")
    p_ad.set_defaults(func=cmd_adapters)

    p_bx = sp.add_parser("build-exog", help="Build exogenous feature table with resources monitoring")
    p_bx.add_argument("--profile", default="local")
    p_bx.add_argument("--env", default="LOCAL")
    p_bx.add_argument("--host", default=settings.db_host)
    p_bx.add_argument("--port", type=int, default=settings.db_port)
    p_bx.add_argument("--user", default=settings.db_user)
    p_bx.set_defaults(password=settings.db_password)
    p_bx.add_argument("--database", default=settings.db_name)
    p_bx.add_argument("--source-schema", default=settings.db_schema)
    p_bx.add_argument("--source-table", default=settings.db_table)
    p_bx.add_argument("--source-where", default=None)
    p_bx.add_argument("--target-schema", default="exog")
    p_bx.add_argument("--target-table", default=f"{settings.db_table}_exog")
    p_bx.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"])
    p_bx.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p_bx.add_argument("--time-col", default=settings.time_col)
    p_bx.add_argument("--target-col", default=settings.target_col)
    p_bx.add_argument("--parallel-workers", type=int, default=4)
    p_bx.add_argument("--enable-gpu-compute", action="store_true")
    p_bx.add_argument("--enable-anomaly-features", action=argparse.BooleanOptionalAction, default=True)
    p_bx.add_argument("--pyod-codegen-yaml", default="./docs/lib_docs/pyod_all_codegen.yaml")
    p_bx.add_argument("--pyod-detectors", default="ECOD,IForest,COPOD")
    p_bx.add_argument("--pyod-contamination", type=float, default=0.1)
    p_bx.add_argument("--anomaly-min-train-size", type=int, default=20)
    p_bx.add_argument("--anomaly-rolling-window", type=int, default=14)
    p_bx.add_argument("--enable-merlion-features", action=argparse.BooleanOptionalAction, default=False)
    p_bx.add_argument(
        "--merlion-codegen-yaml",
        default="./docs/lib_docs/merlion_dashboard_selected_codegen_details.yaml",
    )
    p_bx.add_argument("--merlion-models", default="iforest,lof,spectral_residual,stat_threshold")
    p_bx.add_argument("--merlion-contamination", type=float, default=0.1)
    p_bx.add_argument("--merlion-min-train-size", type=int, default=30)
    p_bx.add_argument("--merlion-n-estimators", type=int, default=100)
    p_bx.add_argument("--merlion-max-n-samples", type=int, default=512)
    p_bx.add_argument("--merlion-random-state", type=int, default=42)
    p_bx.add_argument("--enable-pypots-features", action=argparse.BooleanOptionalAction, default=False)
    p_bx.add_argument("--pypots-codegen-yaml", default="./docs/lib_docs/pypots_all_codegen.yaml")
    p_bx.add_argument("--pypots-models", default="transformer,saits")
    p_bx.add_argument("--pypots-anomaly-rate", type=float, default=0.1)
    p_bx.add_argument("--pypots-window-size", type=int, default=32)
    p_bx.add_argument("--pypots-min-train-windows", type=int, default=20)
    p_bx.add_argument("--pypots-epochs", type=int, default=2)
    p_bx.add_argument("--pypots-batch-size", type=int, default=32)
    p_bx.add_argument("--enable-tsfel-features", action=argparse.BooleanOptionalAction, default=False)
    p_bx.add_argument("--tsfel-codegen-yaml", default="./docs/lib_docs/tsfel_all_codegen.yaml")
    p_bx.add_argument("--tsfel-domains", default="statistical,temporal,spectral")
    p_bx.add_argument("--tsfel-max-features", type=int, default=64)
    p_bx.add_argument("--tsfel-window-size", type=int, default=32)
    p_bx.add_argument("--tsfel-min-train-windows", type=int, default=20)
    p_bx.add_argument("--tsfel-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill")
    p_bx.add_argument("--tsfel-sampling-frequency", type=float, default=1.0)
    p_bx.add_argument("--enable-autogluon-features", action=argparse.BooleanOptionalAction, default=False)
    p_bx.add_argument(
        "--autogluon-codegen-yaml", default="./docs/lib_docs/autogluon__internal__all_codegen.yaml"
    )
    p_bx.add_argument("--autogluon-generators", default="automl_pipeline")
    p_bx.add_argument("--autogluon-window-size", type=int, default=32)
    p_bx.add_argument("--autogluon-min-train-windows", type=int, default=20)
    p_bx.add_argument(
        "--autogluon-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill"
    )
    p_bx.add_argument("--autogluon-max-features", type=int, default=64)
    p_bx.add_argument("--enable-stumpy-features", action=argparse.BooleanOptionalAction, default=False)
    p_bx.add_argument("--stumpy-codegen-yaml", default="./docs/lib_docs/stumpy_all_codegen.yaml")
    p_bx.add_argument("--stumpy-window-size", type=int, default=32)
    p_bx.add_argument("--stumpy-min-train-windows", type=int, default=20)
    p_bx.add_argument(
        "--stumpy-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill"
    )
    p_bx.add_argument("--stumpy-discord-quantile", type=float, default=0.98)
    p_bx.add_argument("--enable-tsfresh-features", action=argparse.BooleanOptionalAction, default=False)
    p_bx.add_argument("--tsfresh-codegen-yaml", default="./docs/lib_docs/tsfresh_all_codegen.yaml")
    p_bx.add_argument("--tsfresh-feature-set", default="minimal")
    p_bx.add_argument("--tsfresh-window-size", type=int, default=32)
    p_bx.add_argument("--tsfresh-min-train-windows", type=int, default=20)
    p_bx.add_argument(
        "--tsfresh-fill-method", choices=["ffill", "bfill", "interpolate", "zero", "mean"], default="ffill"
    )
    p_bx.add_argument("--tsfresh-max-features", type=int, default=64)
    p_bx.add_argument("--tsfresh-n-jobs", type=int, default=0)
    p_bx.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p_bx.add_argument("--lib-docs-dir", default="./docs/lib_docs")
    p_bx.set_defaults(func=cmd_build_exog)

    p_bu = sp.add_parser(
        "build-exog-uni2ts",
        help="Build UNI2TS embedding exogenous table and register resources metrics",
    )
    p_bu.add_argument("--profile", default="local")
    p_bu.add_argument("--env", default="LOCAL")
    p_bu.add_argument("--host", default=settings.db_host)
    p_bu.add_argument("--port", type=int, default=settings.db_port)
    p_bu.add_argument("--user", default=settings.db_user)
    p_bu.set_defaults(password=settings.db_password)
    p_bu.add_argument("--database", default=settings.db_name)
    p_bu.add_argument("--source-schema", default=settings.db_schema)
    p_bu.add_argument("--source-table", default=settings.db_table)
    p_bu.add_argument("--source-where", default=None)
    p_bu.add_argument("--target-schema", default="exog")
    p_bu.add_argument("--target-table", default="uni2ts")
    p_bu.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"])
    p_bu.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p_bu.add_argument("--time-col", default=settings.time_col)
    p_bu.add_argument("--target-col", default=settings.target_col)
    p_bu.add_argument("--context-length", type=int, default=128)
    p_bu.add_argument("--embedding-dim", type=int, default=256)
    p_bu.add_argument("--batch-size", type=int, default=512)
    p_bu.add_argument("--parallel-workers", type=int, default=4)
    p_bu.add_argument("--model-name", default="uni2ts")
    p_bu.add_argument("--model-version", default="2.0.0")
    p_bu.add_argument("--model-checkpoint", default=None)
    p_bu.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p_bu.add_argument("--enable-gpu-compute", action=argparse.BooleanOptionalAction, default=True)
    p_bu.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p_bu.add_argument("--uni2ts-codegen-yaml", default="./docs/lib_docs/uni2ts_all_codegen.yaml")
    p_bu.set_defaults(func=cmd_build_exog_uni2ts)

    p_bt = sp.add_parser(
        "build-exog-timesfm",
        help="Build TimesFM embedding exogenous table and register resources metrics",
    )
    p_bt.add_argument("--profile", default="local")
    p_bt.add_argument("--env", default="LOCAL")
    p_bt.add_argument("--host", default=settings.db_host)
    p_bt.add_argument("--port", type=int, default=settings.db_port)
    p_bt.add_argument("--user", default=settings.db_user)
    p_bt.set_defaults(password=settings.db_password)
    p_bt.add_argument("--database", default=settings.db_name)
    p_bt.add_argument("--source-schema", default=settings.db_schema)
    p_bt.add_argument("--source-table", default=settings.db_table)
    p_bt.add_argument("--source-where", default=None)
    p_bt.add_argument("--ds-start", default=None)
    p_bt.add_argument("--ds-end", default=None)
    p_bt.add_argument("--loto-filter", default=None, help="CSV values")
    p_bt.add_argument("--ts-type-filter", default=None, help="CSV values")
    p_bt.add_argument("--target-schema", default="exog")
    p_bt.add_argument("--target-table", default="timesfm")
    p_bt.add_argument("--if-exists", default="append", choices=["replace", "append", "fail"])
    p_bt.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    p_bt.add_argument("--group-cols", default="loto,ts_type")
    p_bt.add_argument("--time-col", default=settings.time_col)
    p_bt.add_argument("--target-col", default=settings.target_col)
    p_bt.add_argument("--source-row-id-column", default="row_id")
    p_bt.add_argument("--y-idx-order-column", default="row_id")
    p_bt.add_argument(
        "--backend", default="timesfm_forecast_features", choices=["timesfm_forecast_features", "timesfm_transformers"]
    )
    p_bt.add_argument("--model-id", default="google/timesfm-2.5-200m-pytorch")
    p_bt.add_argument("--model-name", default="timesfm")
    p_bt.add_argument("--model-version", default="2.5")
    p_bt.add_argument("--embedding-dim", type=int, default=256)
    p_bt.add_argument("--window-size", type=int, default=128)
    p_bt.add_argument("--min-points", type=int, default=16)
    p_bt.add_argument("--batch-size", type=int, default=64)
    p_bt.add_argument("--normalize-method", default="zscore", choices=["zscore", "none"])
    p_bt.add_argument("--fill-method", default="ffill", choices=["ffill", "zero", "drop"])
    p_bt.add_argument("--parallel-workers", type=int, default=4)
    p_bt.add_argument("--enable-gpu-compute", action=argparse.BooleanOptionalAction, default=True)
    p_bt.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p_bt.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p_bt.add_argument(
        "--postgres-copy-strategy",
        default="binary_copy",
        choices=["csv_buffer", "binary_copy", "psycopg3_row"],
    )
    p_bt.add_argument("--timesfm-codegen-yaml", default="./docs/lib_docs/timesfm_all_codegen.yaml")
    p_bt.set_defaults(func=cmd_build_exog_timesfm)

    p_bc = sp.add_parser(
        "build-exog-chronos",
        help="Build Chronos embedding exogenous table and register resources metrics",
    )
    p_bc.add_argument("--profile", default="local")
    p_bc.add_argument("--env", default="LOCAL")
    p_bc.add_argument("--host", default=settings.db_host)
    p_bc.add_argument("--port", type=int, default=settings.db_port)
    p_bc.add_argument("--user", default=settings.db_user)
    p_bc.set_defaults(password=settings.db_password)
    p_bc.add_argument("--database", default=settings.db_name)
    p_bc.add_argument("--source-schema", default=settings.db_schema)
    p_bc.add_argument("--source-table", default=settings.db_table)
    p_bc.add_argument("--source-where", default=None)
    p_bc.add_argument("--ds-start", default=None)
    p_bc.add_argument("--ds-end", default=None)
    p_bc.add_argument("--loto-filter", default=None, help="CSV values")
    p_bc.add_argument("--ts-type-filter", default=None, help="CSV values")
    p_bc.add_argument("--target-schema", default="exog")
    p_bc.add_argument("--target-table", default="chronos")
    p_bc.add_argument("--if-exists", default="append", choices=["replace", "append", "fail"])
    p_bc.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    p_bc.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p_bc.add_argument("--time-col", default=settings.time_col)
    p_bc.add_argument("--target-col", default=settings.target_col)
    p_bc.add_argument("--source-row-id-column", default="row_id")
    p_bc.add_argument("--y-idx-order-column", default="row_id")
    p_bc.add_argument(
        "--backend",
        default="chronos_pipeline_auto",
        choices=[
            "chronos_pipeline_auto",
            "chronos2_pipeline",
            "chronos_bolt_pipeline",
            "chronos_pipeline_legacy",
            "chronos_forecast_features",
        ],
    )
    p_bc.add_argument("--model-id", default="amazon/chronos-bolt-small")
    p_bc.add_argument("--model-name", default="chronos")
    p_bc.add_argument("--model-version", default="1.0")
    p_bc.add_argument("--embedding-dim", type=int, default=256)
    p_bc.add_argument("--window-size", type=int, default=128)
    p_bc.add_argument("--min-points", type=int, default=16)
    p_bc.add_argument("--batch-size", type=int, default=256)
    p_bc.add_argument("--normalize-method", default="zscore", choices=["zscore", "none"])
    p_bc.add_argument("--fill-method", default="zero", choices=["ffill", "zero", "drop"])
    p_bc.add_argument("--parallel-workers", type=int, default=4)
    p_bc.add_argument("--enable-gpu-compute", action=argparse.BooleanOptionalAction, default=True)
    p_bc.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p_bc.add_argument("--sampling-interval-sec", type=float, default=1.0)
    p_bc.add_argument(
        "--chronos-codegen-yaml",
        default="./docs/lib_docs/chronos-forecasting_scripts_evaluation_agg-relative-score_all_codegen.yaml",
    )
    p_bc.set_defaults(func=cmd_build_exog_chronos)

    p_bu2 = sp.add_parser(
        "build-unified-dataset",
        help="Build unified train dataset from dataset.loto_y_ts + dataset.loto_hist_feat + exog.* tables",
    )
    p_bu2.add_argument("--profile", default="local")
    p_bu2.add_argument("--env", default="LOCAL")
    p_bu2.add_argument("--host", default=settings.db_host)
    p_bu2.add_argument("--port", type=int, default=settings.db_port)
    p_bu2.add_argument("--user", default=settings.db_user)
    p_bu2.set_defaults(password=settings.db_password)
    p_bu2.add_argument("--database", default=settings.db_name)
    p_bu2.add_argument("--base-schema", default=settings.db_schema)
    p_bu2.add_argument("--base-table", default=settings.db_table)
    p_bu2.add_argument("--hist-schema", default=settings.db_schema)
    p_bu2.add_argument("--hist-table", default="loto_hist_feat")
    p_bu2.add_argument("--exog-schema", default="exog")
    p_bu2.add_argument("--id-col", default=settings.id_col)
    p_bu2.add_argument("--time-col", default=settings.time_col)
    p_bu2.add_argument("--target-col", default=settings.target_col)
    p_bu2.add_argument("--key-candidates", default="loto_y_ts_row_id,row_id,unique_id,loto,ts_type,ds")
    p_bu2.add_argument("--include-exog-tables", default=None, help="CSV table names")
    p_bu2.add_argument("--exclude-exog-tables", default=None, help="CSV table names")
    p_bu2.add_argument("--output-schema", default="dataset")
    p_bu2.add_argument("--output-table", default="loto_y_ts_unified")
    p_bu2.add_argument("--output-if-exists", default="replace", choices=["replace", "append", "fail"])
    p_bu2.add_argument("--output-csv-path", default="./artifacts/datasets/loto_y_ts_unified.csv")
    p_bu2.add_argument("--output-parquet-path", default="./artifacts/datasets/loto_y_ts_unified.parquet")
    p_bu2.add_argument("--output-spark-path", default="./artifacts/datasets/loto_y_ts_unified_spark")
    p_bu2.add_argument("--output-spark-format", default="parquet", choices=["parquet", "csv"])
    p_bu2.add_argument("--sort-output", action=argparse.BooleanOptionalAction, default=True)
    p_bu2.add_argument("--create-postgres-index", action=argparse.BooleanOptionalAction, default=True)
    p_bu2.add_argument("--postgres-chunksize", type=int, default=5000)
    p_bu2.add_argument("--postgres-write-mode", default="to_sql", choices=["to_sql", "copy"])
    p_bu2.add_argument("--postgres-copy-chunk-rows", type=int, default=20000)
    p_bu2.add_argument("--fast-mode", action=argparse.BooleanOptionalAction, default=False)
    p_bu2.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    p_bu2.set_defaults(func=cmd_build_unified_dataset)

    p_cg = sp.add_parser(
        "check-unified-grouping",
        help="Check whether unified dataset follows group keys (and group+time uniqueness)",
    )
    p_cg.add_argument("--host", default=settings.db_host)
    p_cg.add_argument("--port", type=int, default=settings.db_port)
    p_cg.add_argument("--user", default=settings.db_user)
    p_cg.set_defaults(password=settings.db_password)
    p_cg.add_argument("--database", default=settings.db_name)
    p_cg.add_argument("--schema", default="dataset")
    p_cg.add_argument("--table", default="loto_y_ts_unified")
    p_cg.add_argument("--group-cols", default="loto,unique_id,ts_type")
    p_cg.add_argument("--time-col", default="ds")
    p_cg.add_argument("--sample-limit", type=int, default=20)
    p_cg.set_defaults(func=cmd_check_unified_grouping)

    p_mc = sp.add_parser(
        "meta-automodel-create",
        help="Create or update one config row in meta.nf_automodel",
    )
    p_mc.add_argument("--config-json", default=None, help="JSON string/file for full config payload")
    p_mc.add_argument("--config-name", default=None)
    p_mc.add_argument("--active", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--priority", type=int, default=None)
    p_mc.add_argument("--base-schema", default=None)
    p_mc.add_argument("--base-table", default=None)
    p_mc.add_argument("--hist-schema", default=None)
    p_mc.add_argument("--hist-table", default=None)
    p_mc.add_argument("--exog-schema", default=None)
    p_mc.add_argument("--output-schema", default=None)
    p_mc.add_argument("--output-table", default=None)
    p_mc.add_argument("--output-if-exists", choices=["replace", "append", "fail"], default=None)
    p_mc.add_argument("--output-csv-path", default=None)
    p_mc.add_argument("--output-parquet-path", default=None)
    p_mc.add_argument("--output-spark-path", default=None)
    p_mc.add_argument("--output-spark-format", choices=["parquet", "csv"], default=None)
    p_mc.add_argument(
        "--unified-filter-json",
        default=None,
        help='JSON object. example={"loto":"bingo5","unique_id":"N1","ts_type":"raw"}',
    )
    p_mc.add_argument(
        "--unified-group-cols-json",
        default=None,
        help='JSON array. default=["loto","unique_id","ts_type"]',
    )
    p_mc.add_argument(
        "--unified-group-validate-strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail each task when grouped key validation fails",
    )
    p_mc.add_argument("--model-name", default=None)
    p_mc.add_argument("--h", type=int, default=None)
    p_mc.add_argument("--auto-cls-model", default=None, help="BaseAuto cls_model override (default: model_name)")
    p_mc.add_argument("--auto-h", type=int, default=None, help="BaseAuto h override (default: --h/horizon)")
    p_mc.add_argument("--auto-loss", default=None)
    p_mc.add_argument("--auto-valid-loss", default=None)
    p_mc.add_argument("--auto-config-json", default=None, help="BaseAuto config JSON (merged before model-params-json)")
    p_mc.add_argument("--auto-search-alg", default=None)
    p_mc.add_argument("--auto-num-samples", type=int, default=None)
    p_mc.add_argument("--auto-cpus", type=int, default=None)
    p_mc.add_argument("--auto-gpus", type=int, default=None)
    p_mc.add_argument("--auto-refit-with-val", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--auto-verbose", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--auto-alias", default=None)
    p_mc.add_argument("--auto-backend", default=None)
    p_mc.add_argument("--auto-callbacks-json", default=None)
    p_mc.add_argument("--model-params-json", default=None)
    p_mc.add_argument("--strict-exog", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--run-cross-validation", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--futr-exog-list-json", default=None, help="JSON string/file -> list[str]")
    p_mc.add_argument("--hist-exog-list-json", default=None, help="JSON string/file -> list[str]")
    p_mc.add_argument("--stat-exog-list-json", default=None, help="JSON string/file -> list[str]")
    p_mc.add_argument("--nf-fit-kwargs-json", default=None, help="JSON object for NeuralForecast.fit kwargs")
    p_mc.add_argument("--nf-predict-kwargs-json", default=None, help="JSON object for NeuralForecast.predict kwargs")
    p_mc.add_argument(
        "--nf-cross-validation-kwargs-json",
        default=None,
        help="JSON object for NeuralForecast.cross_validation kwargs",
    )
    p_mc.add_argument("--nf-save-kwargs-json", default=None, help="JSON object for NeuralForecast.save kwargs")
    p_mc.add_argument("--nf-load-kwargs-json", default=None, help="JSON object for NeuralForecast.load kwargs")
    p_mc.add_argument(
        "--nf-predict-insample-kwargs-json",
        default=None,
        help="JSON object for NeuralForecast.predict_insample kwargs",
    )
    p_mc.add_argument("--param-space-json", default=None)
    p_mc.add_argument(
        "--param-mode-json",
        default=None,
        help='JSON object for fixed/vary flags. ex: {"lr":{"mode":"vary","values":[0.001,0.0005]}}',
    )
    p_mc.add_argument("--random-seed", type=int, default=None)
    p_mc.add_argument("--max-tasks", type=int, default=None)
    p_mc.add_argument("--recursive-depth", type=int, default=None)
    p_mc.add_argument("--run-predict", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--run-evaluate", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--run-explain", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--run-save", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--run-load", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--run-analyze", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--explain-repeats", type=int, default=None)
    p_mc.add_argument("--save-dataset", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--save-overwrite", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--save-path", default=None)
    p_mc.add_argument("--load-check-predict", action=argparse.BooleanOptionalAction, default=None)
    p_mc.add_argument("--note", default=None)
    p_mc.add_argument("--ensure-db-init", action=argparse.BooleanOptionalAction, default=True)
    p_mc.add_argument("--no-upsert-by-name", action="store_true")
    p_mc.set_defaults(func=cmd_meta_automodel_create)

    p_ma = sp.add_parser(
        "meta-automodel-run",
        help="Run exhaustive/recursive AutoModel execution driven by meta.nf_automodel",
    )
    p_ma.add_argument("--config-id", type=int, default=None)
    p_ma.add_argument("--limit", type=int, default=100)
    p_ma.add_argument("--stop-on-error", action="store_true")
    p_ma.add_argument("--ensure-db-init", action=argparse.BooleanOptionalAction, default=True)
    p_ma.add_argument("--skip-existing-success", action=argparse.BooleanOptionalAction, default=True)
    p_ma.add_argument("--allow-failures", action="store_true", help="Return exit code 0 even when failed runs exist")
    p_ma.set_defaults(func=cmd_meta_automodel_run)

    p_mas = sp.add_parser(
        "meta-automodel-arg-spec",
        help="Show argument spec/allowed keys for NeuralForecast AutoModel meta params",
    )
    p_mas.add_argument("--model-name", default="AutoNHITS")
    p_mas.set_defaults(func=cmd_meta_automodel_arg_spec)

    p_mar = sp.add_parser(
        "meta-automodel-report",
        help="Generate deep statistical report/plots from model.nf_automodel",
    )
    p_mar.add_argument("--config-id", type=int, default=None)
    p_mar.add_argument("--run-id", default=None)
    p_mar.add_argument("--status", default=None)
    p_mar.add_argument("--limit", type=int, default=5000)
    p_mar.add_argument("--target-metric", default="metrics.mae")
    p_mar.add_argument("--higher-is-better", action=argparse.BooleanOptionalAction, default=False)
    p_mar.add_argument("--recursive-depth", type=int, default=3)
    p_mar.add_argument("--min-group-size", type=int, default=5)
    p_mar.add_argument("--alpha", type=float, default=0.05)
    p_mar.add_argument("--top-k", type=int, default=20)
    p_mar.add_argument("--out-dir", default=None)
    p_mar.add_argument("--write-outputs", action=argparse.BooleanOptionalAction, default=True)
    p_mar.add_argument("--fail-on-empty", action="store_true")
    p_mar.set_defaults(func=cmd_meta_automodel_report)

    p_sp = sp.add_parser(
        "run-table-pyspark",
        help="Run table read/transform/write pipeline with PySpark via PostgreSQL JDBC",
    )
    p_sp.add_argument("--profile", default="local")
    p_sp.add_argument("--env", default="LOCAL")
    p_sp.add_argument("--host", default=settings.db_host)
    p_sp.add_argument("--port", type=int, default=settings.db_port)
    p_sp.add_argument("--user", default=settings.db_user)
    p_sp.set_defaults(password=settings.db_password)
    p_sp.add_argument("--database", default=settings.db_name)
    p_sp.add_argument("--source-schema", default=settings.db_schema)
    p_sp.add_argument("--source-table", default="loto_y_ts_unified")
    p_sp.add_argument("--source-sql", default=None, help="Optional SQL query as JDBC source (pushdown)")
    p_sp.add_argument("--source-view", default="src_table")
    p_sp.add_argument(
        "--transform-sql",
        default=None,
        help="Spark SQL. use {{source}} placeholder for source view name. simple SELECT * FROM {{source}} WHERE ... is auto-pushdown",
    )
    p_sp.add_argument("--repartition", type=int, default=None)
    p_sp.add_argument("--target-schema", default=settings.exog_schema)
    p_sp.add_argument("--target-table", default=None)
    p_sp.add_argument("--output-if-exists", default="replace", choices=["replace", "append", "fail"])
    p_sp.add_argument("--output-parquet-path", default=None)
    p_sp.add_argument("--output-csv-path", default=None)
    p_sp.add_argument("--output-spark-path", default=None)
    p_sp.add_argument("--output-spark-format", default="parquet", choices=["parquet", "csv"])
    p_sp.add_argument("--spark-master", default=None)
    p_sp.add_argument("--app-name", default="loto_table_runner")
    p_sp.add_argument("--execution-backend", default="auto", choices=["auto", "spark", "pandas", "polars", "dask"])
    p_sp.add_argument("--dask-npartitions", type=int, default=0)
    p_sp.add_argument("--prefer-pandas", action=argparse.BooleanOptionalAction, default=False)
    p_sp.add_argument("--skip-row-count", action=argparse.BooleanOptionalAction, default=True)
    p_sp.add_argument("--spark-ui-enabled", action=argparse.BooleanOptionalAction, default=False)
    p_sp.add_argument("--spark-shuffle-partitions", type=int, default=16)
    p_sp.add_argument("--spark-reader-fetchsize", type=int, default=10000)
    p_sp.add_argument("--postgres-write-mode", default="copy", choices=["copy", "to_sql"])
    p_sp.add_argument("--postgres-copy-chunk-rows", type=int, default=50000)
    p_sp.add_argument("--postgres-lock-timeout-ms", type=int, default=10000)
    p_sp.set_defaults(func=cmd_run_table_pyspark)

    p_sl = sp.add_parser(
        "model-save-load-analyze",
        help="Save/Load/Analyze NeuralForecast model artifacts",
    )
    p_sl.add_argument("--run-id", required=True)
    p_sl.add_argument(
        "--source-path", default=None, help="Default: artifacts/<run_id>. base_dir指定時は base_dir/<run_id> を優先解決"
    )
    p_sl.add_argument(
        "--save-path",
        default=None,
        help="Default: source-path. base_dir指定時は base_dir/<run_id> に解決（{run_id}テンプレート対応）",
    )
    p_sl.add_argument("--run-save", action=argparse.BooleanOptionalAction, default=True)
    p_sl.add_argument("--run-load", action=argparse.BooleanOptionalAction, default=True)
    p_sl.add_argument("--run-analyze", action=argparse.BooleanOptionalAction, default=True)
    p_sl.add_argument("--save-dataset", action=argparse.BooleanOptionalAction, default=False)
    p_sl.add_argument("--save-overwrite", action=argparse.BooleanOptionalAction, default=True)
    p_sl.add_argument("--load-check-predict", action=argparse.BooleanOptionalAction, default=False)
    p_sl.add_argument("--insample-step-size", type=int, default=1)
    p_sl.add_argument("--save-kwargs-json", default=None, help="JSON object for NeuralForecast.save kwargs")
    p_sl.add_argument("--load-kwargs-json", default=None, help="JSON object for NeuralForecast.load kwargs")
    p_sl.add_argument(
        "--predict-insample-kwargs-json",
        default=None,
        help="JSON object for NeuralForecast.predict_insample kwargs",
    )
    p_sl.set_defaults(func=cmd_model_save_load_analyze)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
