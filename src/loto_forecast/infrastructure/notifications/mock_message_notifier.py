from __future__ import annotations

from typing import Any

from loto_forecast.application.notification_events import NotificationChannel, NotificationEvent


class MockMessageNotifier:
    channel = NotificationChannel.MESSAGE

    def __init__(self, *, provider_name: str = "mock-provider") -> None:
        self._provider_name = provider_name

    def is_configured(self) -> bool:
        return False

    def send(self, event: NotificationEvent) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "ok": True,
            "status": "mock",
            "provider": self._provider_name,
            "message": "通知要求は発火したが、外部送信は未設定のため疑似実行です。",
            "event_id": event.event_id,
        }
