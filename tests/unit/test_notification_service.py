from __future__ import annotations

from pathlib import Path

from loto_forecast.application.notification_events import (
    NotificationChannel,
    NotificationEventKind,
    NotificationSeverity,
    build_notification_event,
)
from loto_forecast.application.notification_service import NotificationService, NotificationServiceConfig


class DummyAdapter:
    def __init__(self, channel: NotificationChannel) -> None:
        self.channel = channel
        self.sent = 0

    def is_configured(self) -> bool:
        return True

    def send(self, event):  # type: ignore[no-untyped-def]
        self.sent += 1
        return {"channel": self.channel.value, "ok": True, "status": "sent", "event_id": event.event_id}


def test_notification_service_dedup_and_rate_limit(tmp_path: Path) -> None:
    screen = DummyAdapter(NotificationChannel.SCREEN)
    email = DummyAdapter(NotificationChannel.EMAIL)
    service = NotificationService(
        [screen, email],
        config=NotificationServiceConfig(
            dedup_window_sec=999.0,
            rate_limit_sec=0.0,
            audit_log_path=tmp_path / "notification_audit.jsonl",
        ),
    )
    event = build_notification_event(
        event_id="evt-1",
        kind=NotificationEventKind.OPERATION_START,
        severity=NotificationSeverity.RUNNING,
        title="train start",
        message="running",
        action="train",
        status="running",
    )
    first = service.publish(event)
    second = service.publish(event)

    assert first.suppressed is False
    assert second.suppressed is True
    assert screen.sent == 1
    assert email.sent == 1
    assert (tmp_path / "notification_audit.jsonl").exists()


def test_notification_service_rate_limits_per_channel() -> None:
    screen = DummyAdapter(NotificationChannel.SCREEN)
    service = NotificationService(
        [screen],
        config=NotificationServiceConfig(dedup_window_sec=0.0, rate_limit_sec=999.0),
    )
    event1 = build_notification_event(
        event_id="evt-1",
        kind=NotificationEventKind.OPERATION_START,
        severity=NotificationSeverity.RUNNING,
        title="train start",
        message="running",
        action="train",
        status="running",
    )
    event2 = build_notification_event(
        event_id="evt-2",
        kind=NotificationEventKind.OPERATION_SUCCESS,
        severity=NotificationSeverity.SUCCESS,
        title="train ok",
        message="done",
        action="train",
        status="success",
    )

    first = service.publish(event1)
    second = service.publish(event2)

    assert first.deliveries[0]["status"] == "sent"
    assert second.deliveries[0]["status"] == "rate_limited"
