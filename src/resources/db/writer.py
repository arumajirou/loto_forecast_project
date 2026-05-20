from __future__ import annotations

import json
from typing import Any

from ..config import ResourcesConfig
from ..db.schema import ddl_statements, table_names
from ..metrics import default_metric_defs


class ResourcesDBWriter:
    def __init__(self, cfg: ResourcesConfig):
        self.cfg = cfg
        self.tables = table_names(cfg)

    def _connect(self):
        dsn = self.cfg.to_dsn()
        last_error = None
        try:
            import psycopg

            return psycopg.connect(dsn)
        except Exception as err:
            last_error = err
        try:
            import psycopg2

            return psycopg2.connect(dsn)
        except Exception as err:
            if last_error is not None:
                raise last_error from err
            raise err

    def ensure_schema_and_tables(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                for ddl in ddl_statements(self.cfg):
                    cur.execute(ddl)
            conn.commit()
        self.upsert_metric_def(default_metric_defs())

    def upsert_metric_def(self, defs: list[dict[str, Any]]) -> None:
        if not defs:
            return
        t = self.tables["metric_def"]
        sql = f"""
        INSERT INTO {t} (
          metric_key, scope, unit, description, source_library, source_method, recommended_interval_sec
        ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (metric_key)
        DO UPDATE SET
          scope=EXCLUDED.scope,
          unit=EXCLUDED.unit,
          description=EXCLUDED.description,
          source_library=EXCLUDED.source_library,
          source_method=EXCLUDED.source_method,
          recommended_interval_sec=EXCLUDED.recommended_interval_sec
        """
        vals = [
            (
                d["metric_key"],
                d["scope"],
                d["unit"],
                d.get("description"),
                d.get("source_library"),
                d.get("source_method"),
                d.get("recommended_interval_sec"),
            )
            for d in defs
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, vals)
            conn.commit()

    def insert_run_start(self, row: dict[str, Any]) -> None:
        t = self.tables["run"]
        sql = f"""
        INSERT INTO {t} (
          run_id, started_at, status, env, profile, host, app_name, command, config_hash, tags,
          rows_target, rows_written, rows_failed, error_summary
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
        """
        vals = (
            row["run_id"],
            row["started_at"],
            row.get("status"),
            row.get("env"),
            row.get("profile"),
            row.get("host"),
            row.get("app_name"),
            row.get("command"),
            row.get("config_hash"),
            json.dumps(row.get("tags", {}), ensure_ascii=False),
            row.get("rows_target"),
            row.get("rows_written"),
            row.get("rows_failed"),
            row.get("error_summary"),
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
            conn.commit()

    def update_run_end(self, run_id, row: dict[str, Any]) -> None:
        t = self.tables["run"]
        sql = f"""
        UPDATE {t}
        SET ended_at=%s, status=%s, rows_target=%s, rows_written=%s, rows_failed=%s, error_summary=%s
        WHERE run_id=%s
        """
        vals = (
            row.get("ended_at"),
            row.get("status"),
            row.get("rows_target"),
            row.get("rows_written"),
            row.get("rows_failed"),
            row.get("error_summary"),
            run_id,
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
            conn.commit()

    def insert_span(self, row: dict[str, Any]) -> None:
        t = self.tables["stage"]
        sql = f"""
        INSERT INTO {t} (
          span_id, run_id, parent_span_id, stage_name, function_fqn, file_path,
          started_at, ended_at, duration_ms, batch_no, rows_in, rows_out,
          cpu_user_ms, cpu_system_ms, rss_start_mb, rss_end_mb, rss_peak_mb,
          io_read_bytes_delta, io_write_bytes_delta, net_sent_bytes_delta, net_recv_bytes_delta,
          gpu_util_avg, gpu_mem_used_mb_avg, db_time_ms, db_rows,
          exception_type, exception_msg, extra
        ) VALUES (
          %s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s,%s,%s,
          %s,%s,%s,%s,%s,
          %s,%s,%s,%s,
          %s,%s,%s,%s,
          %s,%s,%s::jsonb
        )
        """
        vals = (
            row["span_id"],
            row["run_id"],
            row.get("parent_span_id"),
            row["stage_name"],
            row.get("function_fqn"),
            row.get("file_path"),
            row["started_at"],
            row["ended_at"],
            row["duration_ms"],
            row.get("batch_no"),
            row.get("rows_in"),
            row.get("rows_out"),
            row.get("cpu_user_ms"),
            row.get("cpu_system_ms"),
            row.get("rss_start_mb"),
            row.get("rss_end_mb"),
            row.get("rss_peak_mb"),
            row.get("io_read_bytes_delta"),
            row.get("io_write_bytes_delta"),
            row.get("net_sent_bytes_delta"),
            row.get("net_recv_bytes_delta"),
            row.get("gpu_util_avg"),
            row.get("gpu_mem_used_mb_avg"),
            row.get("db_time_ms"),
            row.get("db_rows"),
            row.get("exception_type"),
            row.get("exception_msg"),
            json.dumps(row.get("extra", {}), ensure_ascii=False),
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, vals)
            conn.commit()

    def insert_metric_samples(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        t = self.tables["metric"]
        sql = f"""
        INSERT INTO {t} (run_id, span_id, sampled_at, scope, metric_key, metric_value, unit)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        """
        vals = [
            (
                r["run_id"],
                r.get("span_id"),
                r["sampled_at"],
                r["scope"],
                r["metric_key"],
                r["metric_value"],
                r["unit"],
            )
            for r in rows
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, vals)
            conn.commit()
