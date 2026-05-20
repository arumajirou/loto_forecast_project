from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import uuid
from typing import Any

from .collectors import DBCollector, NvmlCollector, PsutilCollector
from .config import ResourcesConfig
from .db.writer import ResourcesDBWriter
from .sampling import ResourceSampler
from .span import ResourceSpan, reset_active_run, set_active_run
from .utils import detect_execution_environment, host_name, now_utc


@dataclasses.dataclass
class ResourceRun:
    cfg: ResourcesConfig
    run_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)

    def __post_init__(self) -> None:
        self.cfg.validate()
        self.db_writer = ResourcesDBWriter(self.cfg)
        self.collectors: list[Any] = []
        self.db_collector = DBCollector()
        self.started_at = now_utc()
        self.current_span_id: str | None = None
        self._stack: list[Any] = []
        self._token = None
        self._sampler = None
        self._rows_target: int | None = None
        self._rows_written: int | None = None
        self._rows_failed: int | None = None
        self._init_collectors()

    def _init_collectors(self) -> None:
        self.collectors.append(PsutilCollector())
        if self.cfg.enable_gpu:
            with contextlib.suppress(Exception):
                self.collectors.append(NvmlCollector())

    def __enter__(self):
        if self.cfg.ensure_schema:
            self.db_writer.ensure_schema_and_tables()
        conf_hash = hashlib.sha256(
            json.dumps(dataclasses.asdict(self.cfg), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        runtime_env = detect_execution_environment()
        tags = dict(self.cfg.tags or {})
        tags["execution_os"] = str(runtime_env.get("execution_os", "unknown"))
        tags["runtime_env"] = runtime_env
        self.db_writer.insert_run_start(
            {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "status": "running",
                "env": self.cfg.env,
                "profile": self.cfg.profile,
                "host": host_name(),
                "app_name": self.cfg.app_name,
                "command": self.cfg.command,
                "config_hash": conf_hash,
                "tags": tags,
                "rows_target": None,
                "rows_written": None,
                "rows_failed": None,
                "error_summary": None,
            }
        )
        self._token = set_active_run(self)
        if self.cfg.enable_sampling:
            self._sampler = ResourceSampler(
                self,
                interval_sec=self.cfg.sampling_interval_sec,
                buffer_size=self.cfg.sample_buffer_size,
            )
            self._sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "success" if exc is None else "failed"
        err = None if exc is None else str(exc)[:400]
        self.end(status=status, error_summary=err)
        return False

    def attach_sqlalchemy_engine(self, engine) -> None:
        self.db_collector.attach_sqlalchemy_engine(engine)
        self.collectors.append(self.db_collector)

    def span(self, stage_name: str, **kwargs):
        return ResourceSpan(self, stage_name=stage_name, **kwargs)

    def set_counts(
        self, rows_target: int | None = None, rows_written: int | None = None, rows_failed: int | None = None
    ):
        self._rows_target = rows_target
        self._rows_written = rows_written
        self._rows_failed = rows_failed

    def end(self, status: str, error_summary: str | None = None) -> None:
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler = None
        self.db_writer.update_run_end(
            self.run_id,
            {
                "ended_at": now_utc(),
                "status": status,
                "rows_target": self._rows_target,
                "rows_written": self._rows_written,
                "rows_failed": self._rows_failed,
                "error_summary": error_summary,
            },
        )
        if self._token is not None:
            reset_active_run(self._token)
            self._token = None

    def _push_span(self, span_id):
        self._stack.append(span_id)
        self.current_span_id = span_id

    def _pop_span(self):
        if self._stack:
            self._stack.pop()
        self.current_span_id = self._stack[-1] if self._stack else None


def start_run(cfg: ResourcesConfig | None = None, **kwargs) -> ResourceRun:
    if cfg is None:
        cfg = ResourcesConfig.from_env()
    if kwargs:
        cfg = dataclasses.replace(cfg, **kwargs)
    return ResourceRun(cfg)
