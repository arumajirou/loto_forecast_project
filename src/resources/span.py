from __future__ import annotations

import contextvars
import functools
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .utils import now_utc, summarize_exception

_ACTIVE_RUN: contextvars.ContextVar[Any] = contextvars.ContextVar("resources_active_run", default=None)


@dataclass
class ResourceSpan:
    run_ctx: Any
    stage_name: str
    function_fqn: str | None = None
    file_path: str | None = None
    batch_no: int | None = None
    rows_in: int | None = None
    rows_out: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.span_id = uuid.uuid4()
        self._start_at = None
        self._end_at = None
        self._start_perf = 0.0
        self._start_snap = {}
        self._parent_span_id = self.run_ctx.current_span_id

    def _snapshot_parallel(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.run_ctx.cfg.parallel_snapshot_workers) as ex:
            fut_map = {ex.submit(c.snapshot): c for c in self.run_ctx.collectors}
            for fut in as_completed(fut_map):
                c = fut_map[fut]
                try:
                    key = getattr(c, "name", c.__class__.__name__.lower())
                    out[key] = fut.result()
                except Exception:
                    continue
        return out

    def __enter__(self):
        self._start_at = now_utc()
        self._start_perf = time.perf_counter()
        self._start_snap = self._snapshot_parallel()
        self.run_ctx._push_span(self.span_id)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end_at = now_utc()
        dur = int((time.perf_counter() - self._start_perf) * 1000.0)
        end_snap = self._snapshot_parallel()

        merged: dict[str, Any] = {}
        for c in self.run_ctx.collectors:
            key = getattr(c, "name", c.__class__.__name__.lower())
            s = self._start_snap.get(key)
            e = end_snap.get(key)
            if not s or not e:
                continue
            try:
                merged.update(c.diff(s, e))
            except Exception:
                continue

        et = None
        em = None
        if exc is not None:
            et, em = summarize_exception(exc)

        row = {
            "span_id": self.span_id,
            "run_id": self.run_ctx.run_id,
            "parent_span_id": self._parent_span_id,
            "stage_name": self.stage_name,
            "function_fqn": self.function_fqn,
            "file_path": self.file_path,
            "started_at": self._start_at,
            "ended_at": self._end_at,
            "duration_ms": dur,
            "batch_no": self.batch_no,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "exception_type": et,
            "exception_msg": em,
            "extra": self.extra,
        }
        row.update(merged)
        self.run_ctx.db_writer.insert_span(row)
        self.run_ctx._pop_span()
        return False


def set_active_run(run_ctx):
    return _ACTIVE_RUN.set(run_ctx)


def reset_active_run(token):
    _ACTIVE_RUN.reset(token)


def get_active_run():
    return _ACTIVE_RUN.get()


def resource_span(stage_name: str, **span_kwargs):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            run = get_active_run()
            if run is None:
                return fn(*args, **kwargs)
            from .utils import resolve_function_identity

            fqn, path = resolve_function_identity(fn)
            with run.span(stage_name=stage_name, function_fqn=fqn, file_path=path, **span_kwargs):
                return fn(*args, **kwargs)

        return wrapper

    return deco
