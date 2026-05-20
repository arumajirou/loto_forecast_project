from __future__ import annotations

from email import message_from_string

from loto_forecast.application.notification_events import (
    NotificationEventKind,
    NotificationSeverity,
    build_notification_event,
)
from loto_forecast.infrastructure.notifications.email_notifier import EmailNotifier


def test_email_notifier_returns_dry_run_preview_when_smtp_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("LF_SMTP_HOST", raising=False)
    monkeypatch.delenv("LF_NOTIFY_FROM", raising=False)
    monkeypatch.delenv("LF_NOTIFY_DRY_RUN", raising=False)
    monkeypatch.delenv("LF_NOTIFY_TO", raising=False)

    notifier = EmailNotifier()
    event = build_notification_event(
        event_id="evt-dry-run",
        kind=NotificationEventKind.OPERATION_FAILURE,
        severity=NotificationSeverity.FAILURE,
        title="train failed",
        message="stderr を確認してください。",
        action="train",
        status="failed",
        command_summary="python -m loto_forecast.cli train --model AutoNHITS",
        error_summary="RuntimeError: boom",
        artifact_paths=["/tmp/example.json"],
        next_actions=["stderr を見る", "設定を修正して再実行する"],
    )

    result = notifier.send(event)
    message = message_from_string(result["preview"])
    body = message.get_payload(decode=True).decode(message.get_content_charset() or "utf-8")

    assert result["status"] == "dry_run"
    assert result["configured"] is False
    assert result["to"] == "zakumagahiyakesita@gmail.com"
    assert "[loto_forecast] train failed" in result["preview"]
    assert "イベント種別: operation_failure" in body
    assert "成功/失敗: failed" in body
    assert "RuntimeError: boom" in body
