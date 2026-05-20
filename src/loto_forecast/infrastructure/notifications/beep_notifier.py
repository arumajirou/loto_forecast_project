from __future__ import annotations

import os
from typing import Any

from loto_forecast.application.notification_events import NotificationChannel, NotificationEvent


class BeepNotifier:
    channel = NotificationChannel.BEEP

    def is_configured(self) -> bool:
        return True

    def send(self, event: NotificationEvent) -> dict[str, Any]:
        enabled = os.getenv("LF_NOTIFY_BEEP_ENABLED", "1").strip().lower() not in {"0", "false", "off"}
        volume = float(os.getenv("LF_NOTIFY_BEEP_VOLUME", "0.35") or 0.35)
        return {
            "channel": self.channel.value,
            "ok": True,
            "status": "queued" if enabled else "disabled",
            "beep": {
                "enabled": enabled,
                "volume": min(1.0, max(0.0, volume)),
                "tone": event.severity.value,
                "fallback_message": event.message,
            },
        }
