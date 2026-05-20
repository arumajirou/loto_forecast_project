"""Local observability helpers for loto_forecast_project.

This package stores browser screenshots, console logs, traces, metrics, and
static diagnostics in local artifact directories. It is intentionally DB-free by
default so it can be used before PostgreSQL is available.
"""

from .store import (
    OBSERVABILITY_ROOT,
    ObservabilityFinding,
    ObservabilitySnapshot,
    build_observability_snapshot,
    classify_event_level,
    detect_duplicate_events,
    load_recent_events,
    record_event,
    summarize_observability,
)

__all__ = [
    "OBSERVABILITY_ROOT",
    "ObservabilityFinding",
    "ObservabilitySnapshot",
    "build_observability_snapshot",
    "classify_event_level",
    "detect_duplicate_events",
    "load_recent_events",
    "record_event",
    "summarize_observability",
]
