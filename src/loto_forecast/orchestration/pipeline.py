from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text

from ..analysis.diagnostics import adf_test, ljung_box
from ..analysis.evaluation import compute_metrics
from ..config.settings import settings
from ..data.db import make_engine, read_timeseries
from ..features.engineering import (
    add_cyclical_time_features,
    add_diff_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    default_feature_config,
    make_future_df,
)
from ..infra.logging_utils import setup_logging
from ..infra.meta_store import (
    ensure_log_tables,
    mark_model_run_end,
    upsert_model_run,
    write_forecast,
    write_log_error_event,
    write_log_run_history,
    write_metrics,
)


def _sanitize_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows usable for model training without dropping on sparse exog columns."""
    required = {settings.id_col, settings.time_col, settings.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}. table must include these")

    out = df.copy()
    out[settings.time_col] = pd.to_datetime(out[settings.time_col], errors="coerce")
    out = out.dropna(subset=[settings.id_col, settings.time_col, settings.target_col])
    out = out.drop_duplicates(subset=[settings.id_col, settings.time_col], keep="last")
    out = out.sort_values([settings.id_col, settings.time_col]).reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "dataset is empty after filtering required columns "
            f"[{settings.id_col}, {settings.time_col}, {settings.target_col}]"
        )
    return out


def prepare_dataset(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    add_cyclical: bool = True,
    add_diff: bool = True,
) -> pd.DataFrame:
    required = {settings.id_col, settings.time_col, settings.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}. table must include these")

    cfg = default_feature_config()
    lags = lags or cfg.lags
    windows = windows or cfg.windows

    out = df.copy()
    out[settings.time_col] = pd.to_datetime(out[settings.time_col])
    out = out.sort_values([settings.id_col, settings.time_col])
    out = add_time_features(out)
    if add_cyclical:
        out = add_cyclical_time_features(out)
    out = add_lag_features(out, lags=lags)
    out = add_rolling_features(out, windows=windows)
    if add_diff:
        out = add_diff_features(out, periods=[1, 7])
    return out


def _read_run_meta(run_id: str) -> dict:
    meta_path = settings.artifact_dir / run_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _to_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return dict(parsed)
            except Exception:
                return {}
    return {}


def _read_retrain_seed_from_db(run_id: str) -> dict[str, Any] | None:
    engine = make_engine()
    # Prefer model.nf_automodel because it stores normalized params_json/horizon/model_name.
    try:
        q_model = text(
            """
            SELECT model_name, horizon, params_json
            FROM model.nf_automodel
            WHERE run_id = :run_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        with engine.begin() as conn:
            row = conn.execute(q_model, {"run_id": str(run_id)}).mappings().first()
        if row:
            return {
                "model_name": str(row.get("model_name") or "").strip(),
                "h": int(row.get("horizon") or settings.default_horizon),
                "model_params": _to_json_dict(row.get("params_json")),
                "source": "model.nf_automodel",
            }
    except Exception:
        pass

    # Fallback: meta.model_run(meta) may still have training metadata even when artifact is missing.
    try:
        q_run = text(
            """
            SELECT model_name, meta
            FROM meta.model_run
            WHERE run_id = :run_id
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
        with engine.begin() as conn:
            row = conn.execute(q_run, {"run_id": str(run_id)}).mappings().first()
        if row:
            meta_obj = _to_json_dict(row.get("meta"))
            model_params = _to_json_dict(meta_obj.get("model_params"))
            h_v = int(meta_obj.get("h") or settings.default_horizon)
            model_name = str(row.get("model_name") or "").strip() or str(meta_obj.get("model_name") or "").strip()
            if model_name:
                return {
                    "model_name": model_name,
                    "h": h_v,
                    "model_params": model_params,
                    "source": "meta.model_run",
                }
    except Exception:
        pass

    # Fallback for meta-automodel run-id style: cfg<config_id>_d<_>_t<_>_YYYYmmdd_HHMMSS
    try:
        m = pd.Series([str(run_id)]).str.extract(r"^cfg(?P<cfg>\d+)_", expand=True)
        cfg_id_raw = str(m.iloc[0, 0]) if not m.empty and m.shape[1] > 0 else ""
        if cfg_id_raw.strip():
            cfg_id = int(cfg_id_raw)
            q_cfg = text(
                """
                SELECT model_name, horizon, model_params_json
                FROM meta.nf_automodel
                WHERE config_id = :config_id
                LIMIT 1
                """
            )
            with engine.begin() as conn:
                row = conn.execute(q_cfg, {"config_id": int(cfg_id)}).mappings().first()
            if row:
                return {
                    "model_name": str(row.get("model_name") or "").strip(),
                    "h": int(row.get("horizon") or settings.default_horizon),
                    "model_params": _to_json_dict(row.get("model_params_json")),
                    "source": "meta.nf_automodel",
                }
    except Exception:
        pass
    return None


def _read_nf_runtime_kwargs(meta: dict | None) -> dict[str, dict]:
    keys = [
        "nf_fit_kwargs",
        "nf_predict_kwargs",
        "nf_cross_validation_kwargs",
        "nf_save_kwargs",
        "nf_load_kwargs",
        "nf_predict_insample_kwargs",
    ]
    payload = {}
    if isinstance(meta, dict):
        raw = meta.get("nf_runtime_kwargs", {})
        payload = raw if isinstance(raw, dict) else {}
    out: dict[str, dict] = {}
    for key in keys:
        value = payload.get(key, {})
        out[key] = dict(value) if isinstance(value, dict) else {}
    return out


def _normalize_dataset_input_method(value: Any, default: str = "db_table") -> str:
    raw = str(value or default).strip().lower()
    allowed = {"db_table", "db_sql", "csv", "parquet", "json"}
    return raw if raw in allowed else str(default)


def _normalize_dataframe_backend(value: Any, default: str = "pandas") -> str:
    raw = str(value or default).strip().lower()
    allowed = {"pandas", "polars", "dask", "spark", "ray"}
    return raw if raw in allowed else str(default)


def _load_dataset_from_source(
    engine,
    *,
    input_method: str,
    dataframe_backend: str,
    dataset_schema: str,
    dataset_table: str,
    dataset_where: str | None,
    dataset_sql: str | None,
    dataset_path: str | None,
) -> tuple[pd.DataFrame, str]:
    method = _normalize_dataset_input_method(input_method, default="db_table")
    backend = _normalize_dataframe_backend(dataframe_backend, default="pandas")
    where_sql = str(dataset_where).strip() if dataset_where is not None else None
    where_sql = where_sql if where_sql else None
    path_obj = Path(str(dataset_path or "")).expanduser() if str(dataset_path or "").strip() else None

    if method == "db_table":
        df = read_timeseries(engine, str(dataset_schema), str(dataset_table), where_sql=where_sql)
        return df, f"{dataset_schema}.{dataset_table}"

    if method == "db_sql":
        sql = str(dataset_sql or "").strip()
        if not sql:
            raise ValueError("dataset_sql is required when dataset_input_method=db_sql")
        with engine.begin() as conn:
            df = pd.read_sql(text(sql), conn)
        return df, "db_sql"

    if path_obj is None:
        raise ValueError(f"dataset_path is required when dataset_input_method={method}")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"dataset_path not found: {path_obj}")

    if where_sql:
        logger.warning("dataset_where is only applied for db_table input; ignored for file input methods")

    def _read_file_with_pandas(method_name: str, path: Path) -> pd.DataFrame:
        if method_name == "csv":
            return pd.read_csv(path)
        if method_name == "parquet":
            return pd.read_parquet(path)
        if method_name == "json":
            try:
                return pd.read_json(path, lines=True)
            except Exception:
                return pd.read_json(path)
        raise ValueError(f"unsupported file input method: {method_name}")

    if backend == "pandas":
        out = _read_file_with_pandas(method, path_obj)
    elif backend == "polars":
        try:
            import polars as pl
        except Exception as e:
            raise RuntimeError(f"dataframe_backend=polars requested but polars is unavailable: {e}") from e
        if method == "csv":
            out = pl.read_csv(path_obj).to_pandas()
        elif method == "parquet":
            out = pl.read_parquet(path_obj).to_pandas()
        elif method == "json":
            try:
                out = pl.read_ndjson(path_obj).to_pandas()
            except Exception:
                out = pl.read_json(path_obj).to_pandas()
        else:
            raise ValueError(f"unsupported file input method: {method}")
    elif backend == "dask":
        try:
            import dask.dataframe as dd
        except Exception as e:
            raise RuntimeError(f"dataframe_backend=dask requested but dask is unavailable: {e}") from e
        if method == "csv":
            out = dd.read_csv(str(path_obj)).compute()
        elif method == "parquet":
            out = dd.read_parquet(str(path_obj)).compute()
        elif method == "json":
            out = dd.read_json(str(path_obj), blocksize=None).compute()
        else:
            raise ValueError(f"unsupported file input method: {method}")
    elif backend == "spark":
        try:
            from pyspark.sql import SparkSession
        except Exception as e:
            raise RuntimeError(f"dataframe_backend=spark requested but pyspark is unavailable: {e}") from e
        spark = (
            SparkSession.getActiveSession()
            or SparkSession.builder.appName("loto_forecast_dataset_loader").getOrCreate()
        )
        if method == "csv":
            sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(path_obj))
        elif method == "parquet":
            sdf = spark.read.parquet(str(path_obj))
        elif method == "json":
            sdf = spark.read.json(str(path_obj))
        else:
            raise ValueError(f"unsupported file input method: {method}")
        out = sdf.toPandas()
    elif backend == "ray":
        try:
            import ray
            import ray.data as rd
        except Exception as e:
            raise RuntimeError(f"dataframe_backend=ray requested but ray is unavailable: {e}") from e
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        if method == "csv":
            ds = rd.read_csv(str(path_obj))
        elif method == "parquet":
            ds = rd.read_parquet(str(path_obj))
        elif method == "json":
            ds = rd.read_json(str(path_obj))
        else:
            raise ValueError(f"unsupported file input method: {method}")
        out = ds.to_pandas()
    else:
        raise ValueError(f"unsupported dataframe_backend: {backend}")

    if settings.time_col in out.columns:
        out[settings.time_col] = pd.to_datetime(out[settings.time_col], errors="coerce")
    return out, f"{method}:{path_obj}"


def train(
    model_name: str,
    h: int,
    model_params: dict | None = None,
    run_id: str | None = None,
    library_name: str = "neuralforecast",
    adapter_name: str = "neuralforecast_auto",
    grid_id: str | None = None,
    task_id: int | None = None,
    dataset_df: pd.DataFrame | None = None,
    dataset_name: str | None = None,
) -> dict:
    from ..models.neuralforecast_model import train_automodel

    run_id = run_id or f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    logfile = setup_logging(run_id)

    raw_model_params = dict(model_params or {})
    train_model_params = dict(raw_model_params)
    train_freq = str(train_model_params.pop("freq", settings.freq) or settings.freq).strip() or str(settings.freq)
    dataset_schema = str(train_model_params.pop("dataset_schema", settings.db_schema) or settings.db_schema)
    dataset_table = str(train_model_params.pop("dataset_table", settings.db_table) or settings.db_table)
    dataset_input_method = _normalize_dataset_input_method(
        train_model_params.pop("dataset_input_method", "db_table"), default="db_table"
    )
    dataset_path = str(train_model_params.pop("dataset_path", "") or "").strip() or None
    dataset_sql = str(train_model_params.pop("dataset_sql", "") or "").strip() or None
    dataframe_backend = _normalize_dataframe_backend(
        train_model_params.pop("dataframe_backend", "pandas"), default="pandas"
    )
    dataset_where = train_model_params.pop("dataset_where", None)
    dataset_where = str(dataset_where).strip() if dataset_where is not None else None
    if dataset_where == "":
        dataset_where = None
    group_by_mode = str(train_model_params.pop("group_by_mode", "loto_unique_id_ts_type") or "loto_unique_id_ts_type")
    target_loto = str(train_model_params.pop("target_loto", "") or "").strip()
    target_unique_id = str(train_model_params.pop("target_unique_id", "") or "").strip()
    target_ts_type = str(train_model_params.pop("target_ts_type", "") or "").strip()
    h_mode = str(train_model_params.pop("h_mode", "fixed") or "fixed")

    def _csv_values(raw: str) -> list[str]:
        return [x.strip() for x in str(raw or "").split(",") if x.strip()]

    target_unique_id_values = _csv_values(target_unique_id)
    if group_by_mode == "loto_ts_type":
        # unique_id is intentionally ignored in this grouping mode.
        target_unique_id = ""
        target_unique_id_values = []
    elif group_by_mode == "loto_unique_id_ts_type" and not target_unique_id_values:
        raise ValueError(
            "group_by_mode=loto_unique_id_ts_type requires non-empty target_unique_id "
            "(学習単位 候補=loto_unique_id_ts_type では unique_id 候補に None/空文字は指定不可)"
        )

    def _apply_in_filter(df: pd.DataFrame, col: str, raw: str) -> pd.DataFrame:
        if col not in df.columns:
            return df
        values = _csv_values(raw)
        if not values:
            return df
        return df[df[col].astype(str).isin(values)].copy()

    engine = make_engine()
    log_schema_ready = False
    try:
        ensure_log_tables(engine)
        log_schema_ready = True
    except Exception as e:
        logger.warning(f"log schema/table ensure failed; continue without log.* write: {e}")

    dataset_label_default = f"{dataset_schema}.{dataset_table}"
    if dataset_df is not None:
        raw = dataset_df.copy()
    else:
        raw, dataset_label_default = _load_dataset_from_source(
            engine,
            input_method=dataset_input_method,
            dataframe_backend=dataframe_backend,
            dataset_schema=dataset_schema,
            dataset_table=dataset_table,
            dataset_where=dataset_where,
            dataset_sql=dataset_sql,
            dataset_path=dataset_path,
        )
    raw = _apply_in_filter(raw, "loto", target_loto)
    raw = _apply_in_filter(raw, settings.id_col, target_unique_id)
    raw = _apply_in_filter(raw, "ts_type", target_ts_type)

    unique_id_count = 0
    if settings.id_col in raw.columns:
        try:
            unique_id_count = int(raw[settings.id_col].dropna().astype(str).nunique())
        except Exception:
            unique_id_count = 0
    if h_mode == "unique_id_count" and unique_id_count > 0:
        h = int(unique_id_count)

    if group_by_mode == "loto_ts_type":
        required_group = {"loto", "ts_type", settings.time_col, settings.target_col}
        if required_group.issubset(set(raw.columns)):
            group_cols = ["loto", "ts_type", settings.time_col]
            agg: dict[str, Any] = {settings.target_col: "mean"}
            for col in raw.columns:
                if col in group_cols or col == settings.target_col or col == settings.id_col:
                    continue
                if pd.api.types.is_numeric_dtype(raw[col]):
                    agg[col] = "mean"
                else:
                    agg[col] = "first"
            raw = raw.groupby(group_cols, as_index=False).agg(agg)
            raw[settings.id_col] = raw["loto"].astype(str) + "__" + raw["ts_type"].astype(str)

    if dataset_df is not None:
        # meta-automodel already passes unified/feature-ready dataframe.
        df = _sanitize_model_input(raw)
    else:
        df = _sanitize_model_input(prepare_dataset(raw))
    dataset_label = dataset_name or dataset_label_default
    data_selection = {
        "dataset_input_method": dataset_input_method,
        "dataframe_backend": dataframe_backend,
        "dataset_schema": dataset_schema,
        "dataset_table": dataset_table,
        "dataset_where": dataset_where,
        "dataset_path": dataset_path,
        "dataset_sql": dataset_sql,
        "group_by_mode": group_by_mode,
        "target_loto": _csv_values(target_loto),
        "target_unique_id": target_unique_id_values,
        "target_ts_type": _csv_values(target_ts_type),
        "h_mode": h_mode,
        "unique_id_count": int(unique_id_count),
    }

    upsert_model_run(
        engine,
        run_id,
        model_name,
        {
            "h": h,
            "freq": train_freq,
            "db_table": dataset_label,
            "model_params": train_model_params,
            "data_selection": data_selection,
        },
        library_name=library_name,
        adapter_name=adapter_name,
        status="running",
        grid_id=grid_id,
        task_id=task_id,
        log_path=str(logfile),
    )
    if log_schema_ready:
        try:
            write_log_run_history(
                engine,
                run_id=run_id,
                event_type="train_start",
                status="running",
                model_name=model_name,
                library_name=library_name,
                adapter_name=adapter_name,
                grid_id=grid_id,
                task_id=task_id,
                horizon=int(h),
                dataset_name=dataset_label,
                log_path=str(logfile),
                message="train started",
                payload={
                    "data_selection": data_selection,
                    "model_params_keys": sorted(list(train_model_params.keys())),
                },
            )
        except Exception as e:
            logger.warning(f"log.run_history write failed(train_start): {e}")

    started_ts = float(time.time())
    try:
        res = train_automodel(
            df=df,
            model_name=model_name,
            h=h,
            freq=train_freq,
            run_id=run_id,
            model_params=train_model_params,
        )
        upsert_model_run(
            engine,
            res.run_id,
            res.model_name,
            {
                "artifact_path": str(res.artifact_path),
                "exog": res.exog,
                "h": h,
                "freq": train_freq,
                "db_table": dataset_label,
                "model_params": train_model_params,
                "data_selection": data_selection,
            },
            library_name=library_name,
            adapter_name=adapter_name,
            status="success",
            grid_id=grid_id,
            task_id=task_id,
            log_path=str(logfile),
        )
        mark_model_run_end(engine, res.run_id, status="success")
        if log_schema_ready:
            try:
                elapsed_sec = float(time.time() - started_ts)
                write_log_run_history(
                    engine,
                    run_id=res.run_id,
                    event_type="train_end",
                    status="success",
                    model_name=res.model_name,
                    library_name=library_name,
                    adapter_name=adapter_name,
                    grid_id=grid_id,
                    task_id=task_id,
                    horizon=int(h),
                    dataset_name=dataset_label,
                    log_path=str(logfile),
                    message="train completed",
                    payload={
                        "elapsed_sec": elapsed_sec,
                        "artifact_path": str(res.artifact_path),
                        "exog": dict(res.exog or {}),
                    },
                )
            except Exception as e:
                logger.warning(f"log.run_history write failed(train_success): {e}")
    except Exception as e:
        mark_model_run_end(engine, run_id, status="failed", error_message=str(e))
        tb = traceback.format_exc()
        if log_schema_ready:
            try:
                elapsed_sec = float(time.time() - started_ts)
                write_log_run_history(
                    engine,
                    run_id=run_id,
                    event_type="train_end",
                    status="failed",
                    model_name=model_name,
                    library_name=library_name,
                    adapter_name=adapter_name,
                    grid_id=grid_id,
                    task_id=task_id,
                    horizon=int(h),
                    dataset_name=dataset_label,
                    log_path=str(logfile),
                    message=str(e),
                    payload={"elapsed_sec": elapsed_sec},
                )
                write_log_error_event(
                    engine,
                    run_id=run_id,
                    model_name=model_name,
                    stage="train",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback_text=tb,
                    payload={
                        "library_name": library_name,
                        "adapter_name": adapter_name,
                        "grid_id": grid_id,
                        "task_id": task_id,
                        "dataset_name": dataset_label,
                    },
                )
            except Exception as log_e:
                logger.warning(f"log.error_event write failed(train_failed): {log_e}")
        logger.error(tb)
        raise

    return {
        "run_id": res.run_id,
        "artifact_path": str(res.artifact_path),
        "model_name": res.model_name,
        "exog": res.exog,
        "log_path": str(logfile),
    }


def retrain(
    base_run_id: str,
    h: int | None = None,
    model_params: dict | None = None,
) -> dict:
    try:
        meta = _read_run_meta(base_run_id)
        model_name = str(meta["model_name"])
        target_h = int(h or meta.get("h") or settings.default_horizon)
        params = model_params or meta.get("model_params") or {}
    except FileNotFoundError as err:
        seed = _read_retrain_seed_from_db(base_run_id)
        if not seed:
            raise FileNotFoundError(
                f"retrain seed not found for run_id={base_run_id}. "
                f"missing artifact meta and no DB record in model.nf_automodel / meta.model_run"
            ) from err
        model_name = str(seed.get("model_name") or "").strip()
        if not model_name:
            raise ValueError(f"retrain seed invalid for run_id={base_run_id}: model_name is empty") from err
        target_h = int(h or seed.get("h") or settings.default_horizon)
        params = model_params or seed.get("model_params") or {}
        logger.warning(f"retrain fallback seed loaded from DB source={seed.get('source')} run_id={base_run_id}")
    new_run_id = f"{base_run_id}_retrain_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    return train(model_name=model_name, h=target_h, model_params=params, run_id=new_run_id)


def predict(
    run_id: str,
    h: int | None = None,
    dataset_df: pd.DataFrame | None = None,
    dataset_input_method: str = "db_table",
    dataset_schema: str | None = None,
    dataset_table: str | None = None,
    dataset_where: str | None = None,
    dataset_sql: str | None = None,
    dataset_path: str | None = None,
    dataframe_backend: str = "pandas",
) -> pd.DataFrame:
    from ..models.neuralforecast_model import load_model, predict_with_model, prepare_nf_frames

    setup_logging(run_id)
    run_dir = settings.artifact_dir / run_id

    meta: dict[str, Any] = {}
    meta_exog = None
    runtime_kwargs = _read_nf_runtime_kwargs(meta)
    try:
        meta = _read_run_meta(run_id)
        meta_exog = meta.get("exog") if isinstance(meta.get("exog"), dict) else None
        runtime_kwargs = _read_nf_runtime_kwargs(meta)
    except Exception:
        meta = {}
        meta_exog = None
        runtime_kwargs = _read_nf_runtime_kwargs(meta)
    predict_freq = str(meta.get("freq") or settings.freq).strip() if isinstance(meta, dict) else str(settings.freq)
    if not predict_freq:
        predict_freq = str(settings.freq)

    load_kwargs = dict(runtime_kwargs.get("nf_load_kwargs", {}))
    predict_kwargs = dict(runtime_kwargs.get("nf_predict_kwargs", {}))
    effective_h = int(h if h is not None else (predict_kwargs.get("h") or settings.default_horizon))
    predict_kwargs["h"] = effective_h

    nf = load_model(run_dir, load_kwargs=load_kwargs)

    engine = make_engine()
    schema_v = str(dataset_schema or settings.db_schema)
    table_v = str(dataset_table or settings.db_table)
    if dataset_df is not None:
        raw = dataset_df.copy()
        df = _sanitize_model_input(raw)
    else:
        raw, _ = _load_dataset_from_source(
            engine,
            input_method=dataset_input_method,
            dataframe_backend=dataframe_backend,
            dataset_schema=schema_v,
            dataset_table=table_v,
            dataset_where=dataset_where,
            dataset_sql=dataset_sql,
            dataset_path=dataset_path,
        )
        df = _sanitize_model_input(prepare_dataset(raw))

    futr = make_future_df(df, h=effective_h, freq=predict_freq)
    futr = add_time_features(futr)
    futr = add_cyclical_time_features(futr)
    df_fit, futr_fit, _ = prepare_nf_frames(df=df, exog=meta_exog, futr_df=futr)

    fcst = predict_with_model(nf, df=df_fit, futr_df=futr_fit, predict_kwargs=predict_kwargs)
    out_path = run_dir / "forecast.parquet"
    fcst.to_parquet(out_path, index=False)
    logger.info(f"forecast saved: {out_path}")

    try:
        write_forecast(engine, run_id, fcst, id_col=settings.id_col, time_col=settings.time_col)
    except Exception as e:
        logger.warning(f"write_forecast failed: {e}")
    return fcst


def predict_with_dataset(run_id: str, dataset_df: pd.DataFrame, h: int | None = None) -> pd.DataFrame:
    from ..models.neuralforecast_model import load_model, predict_with_model, prepare_nf_frames

    setup_logging(run_id)
    run_dir = settings.artifact_dir / run_id

    meta: dict[str, Any] = {}
    meta_exog = None
    runtime_kwargs = _read_nf_runtime_kwargs(meta)
    try:
        meta = _read_run_meta(run_id)
        meta_exog = meta.get("exog") if isinstance(meta.get("exog"), dict) else None
        runtime_kwargs = _read_nf_runtime_kwargs(meta)
    except Exception:
        meta = {}
        meta_exog = None
        runtime_kwargs = _read_nf_runtime_kwargs(meta)
    predict_freq = str(meta.get("freq") or settings.freq).strip() if isinstance(meta, dict) else str(settings.freq)
    if not predict_freq:
        predict_freq = str(settings.freq)

    load_kwargs = dict(runtime_kwargs.get("nf_load_kwargs", {}))
    predict_kwargs = dict(runtime_kwargs.get("nf_predict_kwargs", {}))
    effective_h = int(h if h is not None else (predict_kwargs.get("h") or settings.default_horizon))
    predict_kwargs["h"] = effective_h
    nf = load_model(run_dir, load_kwargs=load_kwargs)

    df = _sanitize_model_input(dataset_df.copy())
    futr = make_future_df(df, h=effective_h, freq=predict_freq)
    futr = add_time_features(futr)
    futr = add_cyclical_time_features(futr)
    df_fit, futr_fit, _ = prepare_nf_frames(df=df, exog=meta_exog, futr_df=futr)

    fcst = predict_with_model(nf, df=df_fit, futr_df=futr_fit, predict_kwargs=predict_kwargs)
    out_path = run_dir / "forecast.parquet"
    fcst.to_parquet(out_path, index=False)
    logger.info(f"forecast saved: {out_path}")
    return fcst


def _build_step_split_metrics(merged: pd.DataFrame, model_col: str, step_eval_size: int) -> list[dict[str, Any]]:
    if merged.empty:
        return []
    step_size = max(1, int(step_eval_size or 1))
    work = merged.copy()
    work = work.sort_values([settings.id_col, settings.time_col]).reset_index(drop=True)
    work["_forecast_step"] = work.groupby(settings.id_col).cumcount() + 1
    if step_size <= 1:
        work["_step_from"] = work["_forecast_step"]
        work["_step_to"] = work["_forecast_step"]
    else:
        work["_step_from"] = ((work["_forecast_step"] - 1) // step_size) * step_size + 1
        work["_step_to"] = work["_step_from"] + step_size - 1

    rows: list[dict[str, Any]] = []
    grouped = work.groupby(["_step_from", "_step_to"], as_index=False)
    for (step_from, step_to), gdf in grouped:
        gdf = gdf.dropna(subset=[settings.target_col, model_col])
        if gdf.empty:
            continue
        m = compute_metrics(gdf[settings.target_col], gdf[model_col])
        metric_values: dict[str, float | None] = {}
        for k, v in dict(m).items():
            try:
                fv = float(v)
            except Exception:
                fv = float("nan")
            metric_values[str(k)] = fv if pd.notna(fv) else None
        rows.append(
            {
                "step_from": int(step_from),
                "step_to": int(min(int(step_to), int(gdf["_forecast_step"].max()))),
                "n": int(len(gdf)),
                **metric_values,
            }
        )
    rows = sorted(rows, key=lambda x: (int(x.get("step_from", 0)), int(x.get("step_to", 0))))
    for row in rows:
        row["step_label"] = (
            str(int(row["step_from"]))
            if int(row["step_from"]) == int(row["step_to"])
            else f"{int(row['step_from'])}-{int(row['step_to'])}"
        )
    return rows


def _normalize_eval_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[settings.id_col] = out[settings.id_col].astype(str)
    out[settings.time_col] = pd.to_datetime(out[settings.time_col], errors="coerce", utc=True).dt.tz_convert(None)
    out = out.dropna(subset=[settings.id_col, settings.time_col])
    out = out.drop_duplicates(subset=[settings.id_col, settings.time_col], keep="last")
    return out.reset_index(drop=True)


def _resolve_forecast_model_col(fcst: pd.DataFrame) -> str:
    excluded = {settings.id_col, settings.time_col, "cutoff", settings.target_col, "y"}
    candidates = [str(c) for c in fcst.columns if str(c) not in excluded]
    if not candidates:
        raise ValueError(f"forecast output has no model columns. columns={list(fcst.columns)}")
    return candidates[0]


def _safe_metric_values(metrics: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, v in dict(metrics).items():
        try:
            fv = float(v)
        except Exception:
            fv = float("nan")
        out[str(k)] = fv if pd.notna(fv) else None
    return out


def _safe_timestamp_str(series: pd.Series) -> str | None:
    if series.empty:
        return None
    ts = pd.to_datetime(series, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None
    return str(ts.iloc[0])


def evaluate(
    run_id: str,
    dataset_df: pd.DataFrame | None = None,
    step_eval_size: int = 1,
    dataset_input_method: str = "db_table",
    dataset_schema: str | None = None,
    dataset_table: str | None = None,
    dataset_where: str | None = None,
    dataset_sql: str | None = None,
    dataset_path: str | None = None,
    dataframe_backend: str = "pandas",
) -> dict:
    from ..models.neuralforecast_model import load_model, predict_with_model, prepare_nf_frames

    setup_logging(run_id)
    run_dir = settings.artifact_dir / run_id
    engine = make_engine()
    if dataset_df is not None:
        raw = dataset_df.copy()
    else:
        raw, _ = _load_dataset_from_source(
            engine,
            input_method=dataset_input_method,
            dataframe_backend=dataframe_backend,
            dataset_schema=str(dataset_schema or settings.db_schema),
            dataset_table=str(dataset_table or settings.db_table),
            dataset_where=dataset_where,
            dataset_sql=dataset_sql,
            dataset_path=dataset_path,
        )
    if dataset_df is not None:
        df = _sanitize_model_input(raw)
    else:
        df = _sanitize_model_input(prepare_dataset(raw))

    meta = _read_run_meta(run_id)
    h = int(meta.get("h", settings.default_horizon))
    eval_freq = str(meta.get("freq") or settings.freq).strip() if isinstance(meta, dict) else str(settings.freq)
    if not eval_freq:
        eval_freq = str(settings.freq)
    runtime_kwargs = _read_nf_runtime_kwargs(meta)
    load_kwargs = dict(runtime_kwargs.get("nf_load_kwargs", {}))
    predict_kwargs = dict(runtime_kwargs.get("nf_predict_kwargs", {}))
    predict_insample_kwargs = dict(runtime_kwargs.get("nf_predict_insample_kwargs", {}))
    predict_kwargs["h"] = h

    df = df.sort_values([settings.id_col, settings.time_col])
    test = df.groupby(settings.id_col).tail(h)
    train_df = df.drop(test.index)

    nf = load_model(run_dir, load_kwargs=load_kwargs)
    model_col: str | None = None
    merged = pd.DataFrame(columns=[settings.id_col, settings.time_col, settings.target_col])
    evaluation_mode = "future_holdout"
    actual_frame = _normalize_eval_key_frame(df[[settings.id_col, settings.time_col, settings.target_col]])

    merge_debug: dict[str, Any] = {
        "actual_rows": int(len(actual_frame)),
        "test_rows": int(len(test)),
        "forecast_rows": 0,
        "merged_rows": 0,
        "actual_ds_min": _safe_timestamp_str(actual_frame[settings.time_col].head(1)),
        "actual_ds_max": _safe_timestamp_str(actual_frame[settings.time_col].tail(1)),
        "forecast_ds_min": None,
        "forecast_ds_max": None,
    }

    try:
        if train_df.empty:
            raise ValueError("train split is empty for holdout evaluation")
        futr = make_future_df(train_df, h=h, freq=eval_freq)
        futr = add_time_features(futr)
        futr = add_cyclical_time_features(futr)
        meta_exog = meta.get("exog") if isinstance(meta.get("exog"), dict) else None
        train_fit, futr_fit, _ = prepare_nf_frames(df=train_df, exog=meta_exog, futr_df=futr)
        fcst = predict_with_model(nf, df=train_fit, futr_df=futr_fit, predict_kwargs=predict_kwargs)

        model_col = _resolve_forecast_model_col(fcst)
        forecast_frame = _normalize_eval_key_frame(fcst[[settings.id_col, settings.time_col, model_col]])
        merged = actual_frame.merge(
            forecast_frame,
            on=[settings.id_col, settings.time_col],
            how="inner",
        )
        merge_debug["forecast_rows"] = int(len(forecast_frame))
        merge_debug["merged_rows"] = int(len(merged))
        merge_debug["forecast_ds_min"] = _safe_timestamp_str(forecast_frame[settings.time_col].head(1))
        merge_debug["forecast_ds_max"] = _safe_timestamp_str(forecast_frame[settings.time_col].tail(1))
    except Exception as e:
        merge_debug["future_holdout_error"] = str(e)

    if merged.empty:
        try:
            insample_kwargs = {"step_size": 1}
            insample_kwargs.update(dict(predict_insample_kwargs))
            insample = nf.predict_insample(**insample_kwargs)
            if isinstance(insample, pd.DataFrame) and (not insample.empty):
                insample_model_col = (
                    model_col
                    if (isinstance(model_col, str) and model_col in insample.columns)
                    else _resolve_forecast_model_col(insample)
                )
                insample_frame = _normalize_eval_key_frame(
                    insample[[settings.id_col, settings.time_col, insample_model_col]]
                )
                merged_ins = actual_frame.merge(
                    insample_frame,
                    on=[settings.id_col, settings.time_col],
                    how="inner",
                )
                if not merged_ins.empty:
                    merged = (
                        merged_ins.sort_values([settings.id_col, settings.time_col])
                        .groupby(settings.id_col, as_index=False)
                        .tail(h)
                        .reset_index(drop=True)
                    )
                    model_col = str(insample_model_col)
                    evaluation_mode = "insample_tail_fallback"
                    merge_debug["insample_rows"] = int(len(insample_frame))
                    merge_debug["insample_merged_rows"] = int(len(merged))
        except Exception as e:
            merge_debug["insample_error"] = str(e)

    merged = merged.sort_values([settings.id_col, settings.time_col]).reset_index(drop=True)
    if not isinstance(model_col, str) or model_col not in merged.columns:
        merged_valid = pd.DataFrame(columns=[settings.id_col, settings.time_col, settings.target_col])
    else:
        merged_valid = merged.dropna(subset=[settings.target_col, model_col]).reset_index(drop=True)
    if merged_valid.empty:
        metrics: dict[str, float | None] = {"mae": None, "rmse": None, "mape": None, "smape": None}
        step_metrics: list[dict[str, Any]] = []
        diag: dict[str, Any] = {
            "adf_residual": None,
            "ljung_box_residual_lag20": None,
        }
    else:
        assert isinstance(model_col, str)
        metrics = _safe_metric_values(compute_metrics(merged_valid[settings.target_col], merged_valid[model_col]))
        step_metrics = _build_step_split_metrics(
            merged_valid,
            model_col=model_col,
            step_eval_size=max(1, int(step_eval_size)),
        )
        residual = merged_valid[settings.target_col] - merged_valid[model_col]
        diag = {"adf_residual": None, "ljung_box_residual_lag20": None}
        try:
            diag["adf_residual"] = adf_test(residual)
        except Exception as e:
            diag["adf_residual"] = {"error": str(e)}
        try:
            diag["ljung_box_residual_lag20"] = ljung_box(residual, lags=20)
        except Exception as e:
            diag["ljung_box_residual_lag20"] = {"error": str(e)}

    out = {
        "metrics": metrics,
        "diagnostics": diag,
        "n_test": int(len(merged_valid)),
        "model_col": model_col,
        "evaluation_mode": evaluation_mode,
        "step_eval_size": int(max(1, int(step_eval_size))),
        "step_metrics": step_metrics,
        "merge_debug": merge_debug,
    }
    try:
        metrics_for_store = {str(k): float(v) for k, v in dict(metrics).items() if v is not None and pd.notna(v)}
        if metrics_for_store:
            write_metrics(engine, run_id, metrics_for_store)
    except Exception as e:
        logger.warning(f"write_metrics failed: {e}")

    (run_dir / "evaluation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"evaluation saved: {run_dir / 'evaluation.json'}")
    return out
