from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any

from loto_forecast.application.notification_events import NotificationChannel, NotificationEvent


class EmailNotifier:
    channel = NotificationChannel.EMAIL

    def __init__(self, *, default_to: str = "zakumagahiyakesita@gmail.com") -> None:
        self._default_to = default_to

    def is_configured(self) -> bool:
        return bool(os.getenv("LF_SMTP_HOST")) and bool(os.getenv("LF_NOTIFY_FROM"))

    def send(self, event: NotificationEvent) -> dict[str, Any]:
        dry_run = os.getenv("LF_NOTIFY_DRY_RUN", "1").strip().lower() not in {"0", "false", "off"}
        to_addr = os.getenv("LF_NOTIFY_TO", self._default_to)
        payload = {
            "channel": self.channel.value,
            "ok": True,
            "status": "dry_run" if dry_run or (not self.is_configured()) else "sent",
            "to": to_addr,
            "configured": self.is_configured(),
        }
        if dry_run or (not self.is_configured()):
            payload["preview"] = self._build_message(event, to_addr).as_string()[:4000]
            return payload

        host = str(os.getenv("LF_SMTP_HOST", "")).strip()
        port = int(os.getenv("LF_SMTP_PORT", "587"))
        username = str(os.getenv("LF_SMTP_USERNAME", "")).strip()
        password = str(os.getenv("LF_SMTP_PASSWORD", "")).strip()
        use_tls = os.getenv("LF_SMTP_USE_TLS", "1").strip().lower() not in {"0", "false", "off"}
        message = self._build_message(event, to_addr)

        with smtplib.SMTP(host, port, timeout=20) as smtp:
            if use_tls:
                smtp.starttls()
            if username:
                smtp.login(username, password)
            smtp.send_message(message)
        return payload

    def _build_message(self, event: NotificationEvent, to_addr: str) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = event.email_subject()
        message["From"] = os.getenv("LF_NOTIFY_FROM", "loto-forecast@localhost")
        message["To"] = to_addr
        body = "\n".join(
            [
                f"イベント種別: {event.kind.value}",
                f"成功/失敗: {event.status}",
                f"実行時刻: {event.created_at.isoformat()}",
                f"実行コマンド要約: {event.command_summary or '-'}",
                f"エラー要約: {event.error_summary or '-'}",
                f"生成物パス: {', '.join(event.artifact_paths) if event.artifact_paths else '-'}",
                f"次の推奨操作: {' / '.join(event.next_actions) if event.next_actions else '-'}",
                "",
                event.message,
            ]
        )
        message.set_content(body)
        return message
