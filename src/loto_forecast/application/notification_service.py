from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .notification_events import NotificationChannel, NotificationEvent

LOGGER = logging.getLogger(__name__)


class NotificationAdapter(Protocol):
    channel: NotificationChannel

    def is_configured(self) -> bool:
        ...

    def send(self, event: NotificationEvent) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class NotificationServiceConfig:
    dedup_window_sec: float = 30.0
    rate_limit_sec: float = 3.0
    audit_log_path: Path | None = None


@dataclass(slots=True)
class NotificationDispatchSummary:
    event_id: str
    suppressed: bool
    reason: str = ""
    deliveries: list[dict[str, Any]] = field(default_factory=list)


class NotificationService:
    def __init__(
        self,
        adapters: list[NotificationAdapter] | tuple[NotificationAdapter, ...],
        *,
        config: NotificationServiceConfig | None = None,
    ) -> None:
        self._adapters = list(adapters)
        self._config = config or NotificationServiceConfig()
        self._last_sent_at_by_key: dict[str, float] = {}
        self._last_sent_at_by_channel: dict[str, float] = {}

    def publish(self, event: NotificationEvent) -> NotificationDispatchSummary:
        now = time.time()
        dedup_key = event.dedup_key()
        last_sent = self._last_sent_at_by_key.get(dedup_key)
        if last_sent is not None and (now - last_sent) < float(self._config.dedup_window_sec):
            summary = NotificationDispatchSummary(
                event_id=event.event_id,
                suppressed=True,
                reason="dedup",
            )
            self._write_audit_row(event, summary)
            return summary

        deliveries: list[dict[str, Any]] = []
        for adapter in self._adapters:
            if adapter.channel not in event.channels:
                continue
            channel_key = adapter.channel.value
            last_channel_sent = self._last_sent_at_by_channel.get(channel_key)
            if last_channel_sent is not None and (now - last_channel_sent) < float(self._config.rate_limit_sec):
                deliveries.append(
                    {
                        "channel": channel_key,
                        "ok": False,
                        "status": "rate_limited",
                    }
                )
                continue
            try:
                result = adapter.send(event)
            except Exception as exc:  # pragma: no cover - safety net
                LOGGER.exception("notification adapter failed: %s", channel_key)
                result = {"channel": channel_key, "ok": False, "status": "failed", "error": str(exc)}
            deliveries.append(result)
            self._last_sent_at_by_channel[channel_key] = now

        self._last_sent_at_by_key[dedup_key] = now
        summary = NotificationDispatchSummary(
            event_id=event.event_id,
            suppressed=False,
            deliveries=deliveries,
        )
        self._write_audit_row(event, summary)
        return summary

    def _write_audit_row(self, event: NotificationEvent, summary: NotificationDispatchSummary) -> None:
        path = self._config.audit_log_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event.to_payload(),
            "dispatch": {
                "event_id": summary.event_id,
                "suppressed": summary.suppressed,
                "reason": summary.reason,
                "deliveries": summary.deliveries,
            },
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
