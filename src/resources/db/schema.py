from __future__ import annotations

from ..config import ResourcesConfig
from ..utils import safe_ident


def _table_base_names() -> dict[str, str]:
    return {
        "run": "run",
        "stage": "stage_span",
        "metric_def": "metric_def",
        "metric": "resource_metric",
    }


def table_names(cfg: ResourcesConfig) -> dict[str, str]:
    schema = safe_ident(cfg.schema)
    base = _table_base_names()
    if cfg.table_naming == "plain":
        names = base
    else:
        ns = safe_ident(cfg.namespace)
        names = {k: f"{ns}_{v}" for k, v in base.items()}
    return {k: f"{schema}.{safe_ident(v)}" for k, v in names.items()}


def ddl_statements(cfg: ResourcesConfig) -> list[str]:
    t = table_names(cfg)
    schema = safe_ident(cfg.schema)
    stage_idx = f"idx_{safe_ident(cfg.namespace)}_stage_run"
    metric_idx = f"idx_{safe_ident(cfg.namespace)}_metric_run"
    if cfg.table_naming == "plain":
        stage_idx = "idx_stage_span_run"
        metric_idx = "idx_resource_metric_run"

    return [
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""
        CREATE TABLE IF NOT EXISTS {t["run"]} (
          run_id UUID PRIMARY KEY,
          started_at TIMESTAMPTZ NOT NULL,
          ended_at TIMESTAMPTZ,
          status TEXT,
          env TEXT,
          profile TEXT,
          host TEXT,
          app_name TEXT,
          command TEXT,
          config_hash TEXT,
          tags JSONB,
          rows_target INTEGER,
          rows_written INTEGER,
          rows_failed INTEGER,
          error_summary TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {t["stage"]} (
          span_id UUID PRIMARY KEY,
          run_id UUID NOT NULL REFERENCES {t["run"]}(run_id),
          parent_span_id UUID,
          stage_name TEXT NOT NULL,
          function_fqn TEXT,
          file_path TEXT,
          started_at TIMESTAMPTZ NOT NULL,
          ended_at TIMESTAMPTZ NOT NULL,
          duration_ms BIGINT NOT NULL,
          batch_no INTEGER,
          rows_in INTEGER,
          rows_out INTEGER,
          cpu_user_ms BIGINT,
          cpu_system_ms BIGINT,
          rss_start_mb REAL,
          rss_end_mb REAL,
          rss_peak_mb REAL,
          io_read_bytes_delta BIGINT,
          io_write_bytes_delta BIGINT,
          net_sent_bytes_delta BIGINT,
          net_recv_bytes_delta BIGINT,
          gpu_util_avg REAL,
          gpu_mem_used_mb_avg REAL,
          db_time_ms BIGINT,
          db_rows BIGINT,
          exception_type TEXT,
          exception_msg TEXT,
          extra JSONB
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {t["metric_def"]} (
          metric_key TEXT PRIMARY KEY,
          scope TEXT NOT NULL,
          unit TEXT NOT NULL,
          description TEXT,
          source_library TEXT,
          source_method TEXT,
          recommended_interval_sec INTEGER
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {t["metric"]} (
          run_id UUID NOT NULL,
          span_id UUID,
          sampled_at TIMESTAMPTZ NOT NULL,
          scope TEXT NOT NULL,
          metric_key TEXT NOT NULL,
          metric_value DOUBLE PRECISION NOT NULL,
          unit TEXT NOT NULL
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {stage_idx} ON {t['stage']}(run_id)",
        f"CREATE INDEX IF NOT EXISTS {metric_idx} ON {t['metric']}(run_id, sampled_at)",
    ]
