from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OBSERVABILITY_ROOT = PROJECT_ROOT / "artifacts" / "observability"
EVENTS_PATH = OBSERVABILITY_ROOT / "events.jsonl"
RUNS_DIR = OBSERVABILITY_ROOT / "browser_runs"
REPORTS_DIR = OBSERVABILITY_ROOT / "reports"

SECRET_PATTERNS = (
    re.compile(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)\s*[:=]\s*['\"]?([^'\"\s]+)"),
    re.compile(r"postgresql://[^:\s]+:[^@\s]+@"),
)


@dataclass(slots=True)
class ObservabilityFinding:
    level: str
    category: str
    message: str
    count: int = 1
    first_seen: str | None = None
    last_seen: str | None = None
    sample: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObservabilitySnapshot:
    generated_at: str
    project_root: str
    events_path: str
    total_events: int
    level_counts: dict[str, int]
    category_counts: dict[str, int]
    duplicate_groups: list[ObservabilityFinding]
    error_findings: list[ObservabilityFinding]
    browser_runs: list[dict[str, Any]]
    screenshot_count: int
    trace_count: int
    log_count: int
    latest_event: dict[str, Any] | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_observability_dirs() -> None:
    OBSERVABILITY_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def mask_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): mask_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_secrets(v) for v in value]
    if not isinstance(value, str):
        return value

    masked = value
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("postgresql"):
            masked = pattern.sub("postgresql://***:***@", masked)
        else:
            masked = pattern.sub(lambda m: f"{m.group(1)}=***", masked)
    return masked


def stable_fingerprint(payload: dict[str, Any]) -> str:
    basis = {
        "source": payload.get("source", ""),
        "category": payload.get("category", ""),
        "level": payload.get("level", ""),
        "message": str(payload.get("message", ""))[:300],
        "exception_type": payload.get("exception_type", ""),
        "path": payload.get("path", ""),
    }
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def classify_event_level(message: str, *, source: str = "", default: str = "INFO") -> str:
    text = f"{source}\n{message}".lower()
    if any(
        token in text for token in ("traceback", "exception", "modulenotfounderror", "error:", " failed", "timeout")
    ):
        return "ERROR"
    if any(token in text for token in ("warning", "warn", "deprecated", "retry")):
        return "WARNING"
    if any(token in text for token in ("success", "passed", "no issues identified", "all checks passed")):
        return "OK"
    return default


def record_event(
    *,
    source: str,
    category: str,
    message: str,
    level: str | None = None,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    ensure_observability_dirs()
    payload = mask_secrets(payload or {})
    event: dict[str, Any] = {
        "ts": utc_now_iso(),
        "source": source,
        "category": category,
        "level": level or classify_event_level(message, source=source),
        "message": mask_secrets(message),
        "run_id": run_id,
        "payload": payload,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    if exc is not None:
        event["exception_type"] = exc.__class__.__name__
        event["traceback"] = mask_secrets("".join(traceback.format_exception(exc)))
    event["fingerprint"] = stable_fingerprint(event)
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return event


def load_recent_events(limit: int = 2000, path: Path | None = None) -> list[dict[str, Any]]:
    path = path or EVENTS_PATH
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[-max(1, int(limit)) :]
    events: list[dict[str, Any]] = []
    for line in selected:
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            events.append(loaded)
    return events


def detect_duplicate_events(events: list[dict[str, Any]], *, min_count: int = 2) -> list[ObservabilityFinding]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        fp = str(event.get("fingerprint") or stable_fingerprint(event))
        grouped.setdefault(fp, []).append(event)

    findings: list[ObservabilityFinding] = []
    for fp, items in grouped.items():
        if len(items) < min_count:
            continue
        first = items[0]
        findings.append(
            ObservabilityFinding(
                level=str(first.get("level", "INFO")),
                category=str(first.get("category", "unknown")),
                message=str(first.get("message", ""))[:500],
                count=len(items),
                first_seen=str(items[0].get("ts", "")),
                last_seen=str(items[-1].get("ts", "")),
                sample={"fingerprint": fp, "source": first.get("source"), "run_id": first.get("run_id")},
            )
        )
    findings.sort(key=lambda item: item.count, reverse=True)
    return findings


def detect_error_findings(events: list[dict[str, Any]]) -> list[ObservabilityFinding]:
    findings: list[ObservabilityFinding] = []
    for event in events:
        level = str(event.get("level", "")).upper()
        message = str(event.get("message", ""))
        if level in {"ERROR", "CRITICAL"} or classify_event_level(message) == "ERROR":
            findings.append(
                ObservabilityFinding(
                    level=level or "ERROR",
                    category=str(event.get("category", "unknown")),
                    message=message[:500],
                    count=1,
                    first_seen=str(event.get("ts", "")),
                    last_seen=str(event.get("ts", "")),
                    sample={
                        "source": event.get("source"),
                        "run_id": event.get("run_id"),
                        "exception_type": event.get("exception_type"),
                        "fingerprint": event.get("fingerprint"),
                    },
                )
            )
    return findings


def _safe_stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def list_browser_runs(limit: int = 50) -> list[dict[str, Any]]:
    ensure_observability_dirs()
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not run_dir.is_dir():
            continue
        manifest = run_dir / "manifest.json"
        payload: dict[str, Any] = {"run_id": run_dir.name, "path": str(run_dir)}
        if manifest.exists():
            try:
                loaded = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload.update(loaded)
            except json.JSONDecodeError:
                payload["manifest_error"] = "invalid json"
        progress_path = run_dir / "progress.jsonl"
        payload["progress_path"] = str(progress_path) if progress_path.exists() else ""
        if progress_path.exists():
            try:
                progress_lines = progress_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if progress_lines:
                    latest_progress = json.loads(progress_lines[-1])
                    if isinstance(latest_progress, dict):
                        payload["progress"] = latest_progress
            except json.JSONDecodeError:
                payload["progress_error"] = "invalid json"
        payload["screenshots"] = len(list(run_dir.rglob("*.png")))
        payload["logs"] = len(list(run_dir.rglob("*.jsonl"))) + len(list(run_dir.rglob("*.log")))
        payload["traces"] = len(list(run_dir.rglob("*.zip"))) + len(list(run_dir.rglob("*.har")))
        runs.append(payload)
        if len(runs) >= limit:
            break
    return runs


def summarize_observability(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    events = events if events is not None else load_recent_events()
    level_counts = Counter(str(e.get("level", "UNKNOWN")).upper() for e in events)
    category_counts = Counter(str(e.get("category", "unknown")) for e in events)
    duplicate_groups = detect_duplicate_events(events)
    error_findings = detect_error_findings(events)
    browser_runs = list_browser_runs()

    screenshot_count = sum(int(run.get("screenshots", 0) or 0) for run in browser_runs)
    trace_count = sum(int(run.get("traces", 0) or 0) for run in browser_runs)
    log_count = sum(int(run.get("logs", 0) or 0) for run in browser_runs)

    return {
        "generated_at": utc_now_iso(),
        "total_events": len(events),
        "level_counts": dict(level_counts),
        "category_counts": dict(category_counts),
        "duplicate_groups": [asdict(item) for item in duplicate_groups[:50]],
        "error_findings": [asdict(item) for item in error_findings[-100:]],
        "browser_runs": browser_runs,
        "screenshot_count": screenshot_count,
        "trace_count": trace_count,
        "log_count": log_count,
        "latest_event": events[-1] if events else None,
    }


def build_observability_snapshot(limit: int = 2000) -> ObservabilitySnapshot:
    events = load_recent_events(limit=limit)
    summary = summarize_observability(events)
    return ObservabilitySnapshot(
        generated_at=str(summary["generated_at"]),
        project_root=str(PROJECT_ROOT),
        events_path=str(EVENTS_PATH),
        total_events=int(summary["total_events"]),
        level_counts=dict(summary["level_counts"]),
        category_counts=dict(summary["category_counts"]),
        duplicate_groups=[ObservabilityFinding(**item) for item in summary["duplicate_groups"]],
        error_findings=[ObservabilityFinding(**item) for item in summary["error_findings"]],
        browser_runs=list(summary["browser_runs"]),
        screenshot_count=int(summary["screenshot_count"]),
        trace_count=int(summary["trace_count"]),
        log_count=int(summary["log_count"]),
        latest_event=summary["latest_event"],
    )


def write_summary_report(path: Path | None = None, *, limit: int = 2000) -> Path:
    ensure_observability_dirs()
    snapshot = build_observability_snapshot(limit=limit)
    path = path or REPORTS_DIR / f"observability_summary_{int(time.time())}.json"
    payload = asdict(snapshot)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
