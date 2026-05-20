from __future__ import annotations

import threading
import time
from typing import Any


class DBCollector:
    """Optional SQLAlchemy event collector for query time/rows/errors."""

    name = "db"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._time_ms = 0.0
        self._rows = 0
        self._errors = 0

    def attach_sqlalchemy_engine(self, engine) -> None:
        from sqlalchemy import event

        @event.listens_for(engine, "before_cursor_execute")
        def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._resmon_start = time.perf_counter()

        @event.listens_for(engine, "after_cursor_execute")
        def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            elapsed = (time.perf_counter() - getattr(context, "_resmon_start", time.perf_counter())) * 1000.0
            with self._lock:
                self._time_ms += elapsed
                self._rows += max(int(getattr(cursor, "rowcount", 0)), 0)

        @event.listens_for(engine, "handle_error")
        def _handle_error(exception_context):
            with self._lock:
                self._errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"db_time_ms": float(self._time_ms), "db_rows": int(self._rows), "db_errors": int(self._errors)}

    def diff(self, start: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
        return {
            "db_time_ms": int(end["db_time_ms"] - start["db_time_ms"]),
            "db_rows": int(end["db_rows"] - start["db_rows"]),
            "db_errors": int(end["db_errors"] - start["db_errors"]),
        }

    def sample_metrics(self, snap: dict[str, Any]) -> list[tuple[str, float, str, str]]:
        return [
            ("db.query_time_ms_total", float(snap.get("db_time_ms", 0.0)), "ms", "db"),
        ]
