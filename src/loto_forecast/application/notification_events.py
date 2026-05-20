from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NotificationSeverity(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    RUNNING = "running"


class NotificationEventKind(str, Enum):
    ACTION_CONFIRMED = "action_confirmed"
    OPERATION_START = "operation_start"
    OPERATION_SUCCESS = "operation_success"
    OPERATION_FAILURE = "operation_failure"
    LONG_RUNNING_START = "long_running_start"
    LONG_RUNNING_COMPLETE = "long_running_complete"
    EXCEPTION = "exception"


class NotificationChannel(str, Enum):
    SCREEN = "screen"
    BEEP = "beep"
    EMAIL = "email"
    MESSAGE = "message"


@dataclass(slots=True)
class NotificationEvent:
    event_id: str
    kind: NotificationEventKind
    severity: NotificationSeverity
    title: str
    message: str
    action: str
    status: str
    command_summary: str = ""
    error_summary: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    channels: tuple[NotificationChannel, ...] = (
        NotificationChannel.SCREEN,
        NotificationChannel.BEEP,
        NotificationChannel.EMAIL,
        NotificationChannel.MESSAGE,
    )
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def dedup_key(self) -> str:
        command = self.command_summary.strip() or str(self.metadata.get("command", "")).strip()
        return "|".join(
            [
                self.kind.value,
                self.severity.value,
                self.action.strip().lower(),
                self.status.strip().lower(),
                command.lower(),
                self.error_summary.strip().lower(),
            ]
        )

    def email_subject(self) -> str:
        return f"[loto_forecast] {self.title}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "action": self.action,
            "status": self.status,
            "command_summary": self.command_summary,
            "error_summary": self.error_summary,
            "artifact_paths": list(self.artifact_paths),
            "next_actions": list(self.next_actions),
            "metadata": dict(self.metadata),
            "channels": [channel.value for channel in self.channels],
            "created_at": self.created_at.isoformat(),
        }


def build_notification_event(
    *,
    event_id: str,
    kind: NotificationEventKind,
    severity: NotificationSeverity,
    title: str,
    message: str,
    action: str,
    status: str,
    command_summary: str = "",
    error_summary: str = "",
    artifact_paths: list[str] | None = None,
    next_actions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    channels: tuple[NotificationChannel, ...] | None = None,
) -> NotificationEvent:
    return NotificationEvent(
        event_id=event_id,
        kind=kind,
        severity=severity,
        title=title,
        message=message,
        action=action,
        status=status,
        command_summary=command_summary,
        error_summary=error_summary,
        artifact_paths=list(artifact_paths or []),
        next_actions=list(next_actions or []),
        metadata=dict(metadata or {}),
        channels=channels
        or (
            NotificationChannel.SCREEN,
            NotificationChannel.BEEP,
            NotificationChannel.EMAIL,
            NotificationChannel.MESSAGE,
        ),
    )
