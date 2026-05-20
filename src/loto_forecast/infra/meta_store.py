from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..config.settings import settings


def _schema(schema: str | None = None) -> str:
    """Backward-compatible metadata schema default.

    The source `dataset` schema is read-only. New metadata/result writes should
    use one of the typed helpers below instead of falling back to dataset.
    """
    return schema or settings.meta_schema


def _model_schema(schema: str | None = None) -> str:
    return schema or settings.model_schema


def _resources_schema(schema: str | None = None) -> str:
    return schema or settings.resources_schema


def _log_schema(schema: str | None = None) -> str:
    return schema or settings.log_schema


def upsert_model_run(
    engine: Engine,
    run_id: str,
    model_name: str,
    meta: dict,
    library_name: str = "neuralforecast",
    adapter_name: str = "neuralforecast_auto",
    status: str = "running",
    grid_id: str | None = None,
    task_id: int | None = None,
    log_path: str | None = None,
    schema: str | None = None,
) -> None:
    sch = _schema(schema)
    q = text(
        f"""
    INSERT INTO {sch}.model_run
      (run_id, model_name, meta, library_name, adapter_name, status, grid_id, task_id, log_path)
    VALUES
      (:run_id, :model_name, CAST(:meta AS jsonb), :library_name, :adapter_name, :status, :grid_id, :task_id, :log_path)
    ON CONFLICT (run_id) DO UPDATE SET
      model_name = EXCLUDED.model_name,
      meta = EXCLUDED.meta,
      library_name = EXCLUDED.library_name,
      adapter_name = EXCLUDED.adapter_name,
      status = EXCLUDED.status,
      grid_id = EXCLUDED.grid_id,
      task_id = EXCLUDED.task_id,
      log_path = EXCLUDED.log_path;
    """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "run_id": run_id,
                "model_name": model_name,
                "meta": json.dumps(meta, ensure_ascii=False),
                "library_name": library_name,
                "adapter_name": adapter_name,
                "status": status,
                "grid_id": grid_id,
                "task_id": task_id,
                "log_path": log_path,
            },
        )


def mark_model_run_end(
    engine: Engine,
    run_id: str,
    status: str = "success",
    error_message: str | None = None,
    schema: str | None = None,
) -> None:
    sch = _schema(schema)
    q = text(
        f"UPDATE {sch}.model_run SET ended_at = now(), status=:status, error_message=:error_message WHERE run_id=:run_id"
    )
    with engine.begin() as conn:
        conn.execute(q, {"run_id": run_id, "status": status, "error_message": error_message})


def write_metrics(engine: Engine, run_id: str, metrics: dict, schema: str | None = None) -> None:
    sch = _model_schema(schema)
    q = text(
        f"""
    INSERT INTO {sch}.model_metric (run_id, metric_name, metric_value)
    VALUES (:run_id, :metric_name, :metric_value)
    ON CONFLICT (run_id, metric_name) DO UPDATE
      SET metric_value = EXCLUDED.metric_value, created_at = now();
    """
    )
    with engine.begin() as conn:
        for k, v in metrics.items():
            conn.execute(q, {"run_id": run_id, "metric_name": k, "metric_value": float(v)})


def write_forecast(
    engine: Engine,
    run_id: str,
    fcst: pd.DataFrame,
    id_col: str = "unique_id",
    time_col: str = "ds",
    schema: str | None = None,
) -> None:
    model_cols = [c for c in fcst.columns if c not in (id_col, time_col)]
    if not model_cols:
        return
    yhat_col = model_cols[0]
    out = fcst[[id_col, time_col, yhat_col]].copy()
    out = out.rename(columns={yhat_col: "yhat"})
    out["run_id"] = run_id
    out.to_sql(
        "forecast", engine, schema=_model_schema(schema), if_exists="append", index=False, method="multi", chunksize=5000
    )


def write_exog_contribution(engine: Engine, run_id: str, imp: pd.DataFrame, schema: str | None = None) -> None:
    if imp is None or imp.empty:
        return
    out = imp.copy()
    if "feature" in out.columns:
        out = out.rename(columns={"feature": "feature_name"})
    if "delta_mae_mean" in out.columns:
        out = out.rename(columns={"delta_mae_mean": "importance"})
    if "method" not in out.columns:
        out["method"] = "unknown"
    out["run_id"] = run_id
    out = out[["run_id", "feature_name", "importance", "method"]]
    out.to_sql(
        "exog_contribution",
        engine,
        schema=_model_schema(schema),
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )


def write_resource_samples(
    engine: Engine, run_id: str, samples: list[dict[str, Any]], schema: str | None = None
) -> None:
    if not samples:
        return
    sch = _resources_schema(schema)
    ensure_resource_sample_columns(engine, schema=sch)
    q = text(
        f"""
    INSERT INTO {sch}.resource_sample (
      run_id, ts, cpu_percent, mem_percent, rss_mb,
      process_cpu_percent, system_cpu_percent, gpu_util, gpu_mem_mb, gpu_name, pid
    )
    VALUES (
      :run_id, :ts, :cpu_percent, :mem_percent, :rss_mb,
      :process_cpu_percent, :system_cpu_percent, :gpu_util, :gpu_mem_mb, :gpu_name, :pid
    )
    """
    )
    with engine.begin() as conn:
        for s in samples:
            ts = s.get("ts")
            if isinstance(ts, (float, int)):
                ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            conn.execute(
                q,
                {
                    "run_id": run_id,
                    "ts": ts,
                    "cpu_percent": float(s.get("cpu_percent", s.get("system_cpu_percent", 0.0))),
                    "mem_percent": float(s.get("mem_percent", 0.0)),
                    "rss_mb": float(s.get("rss_mb", 0.0)),
                    "process_cpu_percent": float(s.get("process_cpu_percent", 0.0)),
                    "system_cpu_percent": float(s.get("system_cpu_percent", s.get("cpu_percent", 0.0))),
                    "gpu_util": float(s["gpu_util"]) if s.get("gpu_util") is not None else None,
                    "gpu_mem_mb": float(s["gpu_mem_mb"]) if s.get("gpu_mem_mb") is not None else None,
                    "gpu_name": s.get("gpu_name"),
                    "pid": int(s["pid"]) if s.get("pid") is not None else None,
                },
            )


def ensure_resource_sample_columns(engine: Engine, schema: str | None = None) -> None:
    sch = _resources_schema(schema)
    ddl_list = [
        f"CREATE SCHEMA IF NOT EXISTS {sch}",
        f'ALTER TABLE IF EXISTS {sch}.resource_sample ADD COLUMN IF NOT EXISTS process_cpu_percent DOUBLE PRECISION',
        f'ALTER TABLE IF EXISTS {sch}.resource_sample ADD COLUMN IF NOT EXISTS system_cpu_percent DOUBLE PRECISION',
        f'ALTER TABLE IF EXISTS {sch}.resource_sample ADD COLUMN IF NOT EXISTS gpu_util DOUBLE PRECISION',
        f'ALTER TABLE IF EXISTS {sch}.resource_sample ADD COLUMN IF NOT EXISTS gpu_mem_mb DOUBLE PRECISION',
        f'ALTER TABLE IF EXISTS {sch}.resource_sample ADD COLUMN IF NOT EXISTS gpu_name TEXT',
        f'ALTER TABLE IF EXISTS {sch}.resource_sample ADD COLUMN IF NOT EXISTS pid INTEGER',
    ]
    with engine.begin() as conn:
        for ddl in ddl_list:
            conn.execute(text(ddl))


def create_grid_definition(
    engine: Engine,
    grid_id: str,
    library_name: str,
    adapter_name: str,
    model_name: str,
    horizon: int,
    param_space: dict,
    exog_policy: dict | None = None,
    run_predict: bool = True,
    run_evaluate: bool = True,
    max_tasks: int | None = None,
    note: str | None = None,
    created_by: str | None = None,
    schema: str | None = None,
) -> None:
    sch = _schema(schema)
    q = text(
        f"""
    INSERT INTO {sch}.grid_search_definition (
      grid_id, library_name, adapter_name, model_name, horizon,
      param_space, exog_policy, run_predict, run_evaluate, max_tasks,
      note, created_by
    ) VALUES (
      :grid_id, :library_name, :adapter_name, :model_name, :horizon,
      CAST(:param_space AS jsonb), CAST(:exog_policy AS jsonb), :run_predict, :run_evaluate, :max_tasks,
      :note, :created_by
    )
    ON CONFLICT (grid_id) DO UPDATE SET
      library_name = EXCLUDED.library_name,
      adapter_name = EXCLUDED.adapter_name,
      model_name = EXCLUDED.model_name,
      horizon = EXCLUDED.horizon,
      param_space = EXCLUDED.param_space,
      exog_policy = EXCLUDED.exog_policy,
      run_predict = EXCLUDED.run_predict,
      run_evaluate = EXCLUDED.run_evaluate,
      max_tasks = EXCLUDED.max_tasks,
      note = EXCLUDED.note,
      created_by = EXCLUDED.created_by,
      created_at = now();
    """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "grid_id": grid_id,
                "library_name": library_name,
                "adapter_name": adapter_name,
                "model_name": model_name,
                "horizon": int(horizon),
                "param_space": json.dumps(param_space, ensure_ascii=False),
                "exog_policy": json.dumps(exog_policy or {}, ensure_ascii=False),
                "run_predict": bool(run_predict),
                "run_evaluate": bool(run_evaluate),
                "max_tasks": max_tasks,
                "note": note,
                "created_by": created_by,
            },
        )


def replace_grid_tasks(
    engine: Engine,
    grid_id: str,
    tasks: list[dict[str, Any]],
    schema: str | None = None,
) -> None:
    sch = _schema(schema)
    q_delete = text(f"DELETE FROM {sch}.grid_search_task WHERE grid_id=:grid_id")
    q_insert = text(
        f"""
    INSERT INTO {sch}.grid_search_task (grid_id, task_order, param_values, status)
    VALUES (:grid_id, :task_order, CAST(:param_values AS jsonb), 'pending')
    """
    )
    with engine.begin() as conn:
        conn.execute(q_delete, {"grid_id": grid_id})
        for idx, task_params in enumerate(tasks, start=1):
            conn.execute(
                q_insert,
                {
                    "grid_id": grid_id,
                    "task_order": idx,
                    "param_values": json.dumps(task_params, ensure_ascii=False),
                },
            )


def get_grid_definition(engine: Engine, grid_id: str, schema: str | None = None) -> dict[str, Any] | None:
    sch = _schema(schema)
    q = text(f"SELECT * FROM {sch}.grid_search_definition WHERE grid_id=:grid_id")
    with engine.connect() as conn:
        row = conn.execute(q, {"grid_id": grid_id}).mappings().first()
    return dict(row) if row else None


def list_grid_tasks(
    engine: Engine,
    grid_id: str,
    status: str | None = None,
    limit: int = 1000,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    sch = _schema(schema)
    if status:
        q = text(
            f"""
        SELECT * FROM {sch}.grid_search_task
        WHERE grid_id=:grid_id AND status=:status
        ORDER BY task_order
        LIMIT :limit
        """
        )
        params = {"grid_id": grid_id, "status": status, "limit": int(limit)}
    else:
        q = text(
            f"""
        SELECT * FROM {sch}.grid_search_task
        WHERE grid_id=:grid_id
        ORDER BY task_order
        LIMIT :limit
        """
        )
        params = {"grid_id": grid_id, "limit": int(limit)}

    with engine.connect() as conn:
        rows = conn.execute(q, params).mappings().all()
    return [dict(r) for r in rows]


def start_grid_task(
    engine: Engine,
    task_id: int,
    run_id: str,
    log_path: str | None = None,
    schema: str | None = None,
) -> None:
    sch = _schema(schema)
    q = text(
        f"""
    UPDATE {sch}.grid_search_task
    SET status='running', run_id=:run_id, started_at=now(), updated_at=now(), log_path=:log_path
    WHERE task_id=:task_id
    """
    )
    with engine.begin() as conn:
        conn.execute(q, {"task_id": int(task_id), "run_id": run_id, "log_path": log_path})


def finish_grid_task(
    engine: Engine,
    task_id: int,
    status: str,
    result: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    resource_summary: dict[str, Any] | None,
    error_message: str | None = None,
    schema: str | None = None,
) -> None:
    sch = _schema(schema)
    q = text(
        f"""
    UPDATE {sch}.grid_search_task
    SET status=:status,
        ended_at=now(),
        updated_at=now(),
        result=CAST(:result AS jsonb),
        metrics=CAST(:metrics AS jsonb),
        resource_summary=CAST(:resource_summary AS jsonb),
        error_message=:error_message
    WHERE task_id=:task_id
    """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "task_id": int(task_id),
                "status": status,
                "result": json.dumps(result or {}, ensure_ascii=False),
                "metrics": json.dumps(metrics or {}, ensure_ascii=False),
                "resource_summary": json.dumps(resource_summary or {}, ensure_ascii=False),
                "error_message": error_message,
            },
        )


def log_execution_event(
    engine: Engine,
    event_type: str,
    message: str,
    task_id: int | None = None,
    run_id: str | None = None,
    level: str = "INFO",
    payload: dict[str, Any] | None = None,
    schema: str | None = None,
) -> None:
    sch = _log_schema(schema)
    q = text(
        f"""
    INSERT INTO {sch}.execution_event_log (task_id, run_id, level, event_type, message, payload)
    VALUES (:task_id, :run_id, :level, :event_type, :message, CAST(:payload AS jsonb))
    """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "task_id": task_id,
                "run_id": run_id,
                "level": level,
                "event_type": event_type,
                "message": message,
                "payload": json.dumps(payload or {}, ensure_ascii=False),
            },
        )


def ensure_log_tables(engine: Engine, schema: str = "log") -> None:
    sch = str(schema or "log").strip() or "log"
    ddl_list = [
        f"CREATE SCHEMA IF NOT EXISTS {sch}",
        f"""
        CREATE TABLE IF NOT EXISTS {sch}.run_history (
          history_id BIGSERIAL PRIMARY KEY,
          run_id TEXT NOT NULL,
          event_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
          event_type TEXT NOT NULL,
          status TEXT,
          model_name TEXT,
          library_name TEXT,
          adapter_name TEXT,
          grid_id TEXT,
          task_id BIGINT,
          horizon INTEGER,
          dataset_name TEXT,
          log_path TEXT,
          message TEXT,
          payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {sch}.error_event (
          error_id BIGSERIAL PRIMARY KEY,
          run_id TEXT,
          event_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
          model_name TEXT,
          stage TEXT,
          error_type TEXT,
          error_message TEXT,
          traceback TEXT,
          payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        )
        """,
        f"CREATE INDEX IF NOT EXISTS idx_{sch}_run_history_run_ts ON {sch}.run_history(run_id, event_ts DESC)",
        f"CREATE INDEX IF NOT EXISTS idx_{sch}_run_history_status ON {sch}.run_history(status, event_ts DESC)",
        f"CREATE INDEX IF NOT EXISTS idx_{sch}_error_event_run_ts ON {sch}.error_event(run_id, event_ts DESC)",
    ]
    with engine.begin() as conn:
        for ddl in ddl_list:
            conn.execute(text(ddl))


def write_log_run_history(
    engine: Engine,
    *,
    run_id: str,
    event_type: str,
    status: str | None = None,
    model_name: str | None = None,
    library_name: str | None = None,
    adapter_name: str | None = None,
    grid_id: str | None = None,
    task_id: int | None = None,
    horizon: int | None = None,
    dataset_name: str | None = None,
    log_path: str | None = None,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    schema: str = "log",
) -> None:
    ensure_log_tables(engine, schema=schema)
    sch = str(schema or "log").strip() or "log"
    q = text(
        f"""
        INSERT INTO {sch}.run_history (
          run_id, event_type, status, model_name, library_name, adapter_name,
          grid_id, task_id, horizon, dataset_name, log_path, message, payload
        )
        VALUES (
          :run_id, :event_type, :status, :model_name, :library_name, :adapter_name,
          :grid_id, :task_id, :horizon, :dataset_name, :log_path, :message, CAST(:payload AS jsonb)
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "run_id": str(run_id),
                "event_type": str(event_type),
                "status": status,
                "model_name": model_name,
                "library_name": library_name,
                "adapter_name": adapter_name,
                "grid_id": grid_id,
                "task_id": int(task_id) if task_id is not None else None,
                "horizon": int(horizon) if horizon is not None else None,
                "dataset_name": dataset_name,
                "log_path": log_path,
                "message": message,
                "payload": json.dumps(payload or {}, ensure_ascii=False),
            },
        )


def write_log_error_event(
    engine: Engine,
    *,
    run_id: str | None,
    model_name: str | None,
    stage: str,
    error_type: str | None,
    error_message: str,
    traceback_text: str | None = None,
    payload: dict[str, Any] | None = None,
    schema: str = "log",
) -> None:
    ensure_log_tables(engine, schema=schema)
    sch = str(schema or "log").strip() or "log"
    q = text(
        f"""
        INSERT INTO {sch}.error_event (
          run_id, model_name, stage, error_type, error_message, traceback, payload
        )
        VALUES (
          :run_id, :model_name, :stage, :error_type, :error_message, :traceback, CAST(:payload AS jsonb)
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(
            q,
            {
                "run_id": str(run_id) if run_id is not None else None,
                "model_name": model_name,
                "stage": str(stage),
                "error_type": error_type,
                "error_message": str(error_message),
                "traceback": traceback_text,
                "payload": json.dumps(payload or {}, ensure_ascii=False),
            },
        )
