from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from loto_forecast.application.notification_events import NotificationChannel, NotificationEvent

from .dialogs import render_notification_dialog

SESSION_QUEUE_KEY = "ui_notification_queue"


class StreamlitNotificationAdapter:
    channel = NotificationChannel.SCREEN

    def is_configured(self) -> bool:
        return True

    def send(self, event: NotificationEvent) -> dict[str, Any]:
        return {"channel": self.channel.value, "ok": True, "status": "deferred"}


def drain_notifications() -> list[dict[str, Any]]:
    queue = list(st.session_state.get(SESSION_QUEUE_KEY, []))
    st.session_state[SESSION_QUEUE_KEY] = []
    return queue


def render_notification_center(events: Iterable[dict[str, Any]]) -> None:
    rendered: list[dict[str, Any]] = list(events)
    if not rendered:
        return
    latest = rendered[-1]
    render_notification_dialog(latest)
    for event in rendered:
        severity = str(event.get("severity", "running"))
        toast = getattr(st, "toast", None)
        text = f"{event.get('title', '通知')}: {event.get('message', '')}"
        if callable(toast):
            toast(text, icon=_severity_icon(severity))
        else:
            _legacy_inline_notice(severity, text)
    _render_beep_script(rendered)


def _severity_icon(severity: str) -> str:
    return {
        "success": "✅",
        "failure": "❌",
        "warning": "⚠️",
        "running": "⏳",
    }.get(severity, "ℹ️")


def _legacy_inline_notice(severity: str, text: str) -> None:
    renderer = {
        "success": st.success,
        "failure": st.error,
        "warning": st.warning,
        "running": st.info,
    }.get(severity, st.info)
    renderer(text)


def _render_beep_script(events: list[dict[str, Any]]) -> None:
    beep_events = []
    for event in events:
        beep_cfg = event.get("metadata", {}).get("beep")
        if isinstance(beep_cfg, dict):
            beep_events.append(beep_cfg)
    if not beep_events:
        return
    script = f"""
    <script>
    const events = {json.dumps(beep_events)};
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    for (const event of events) {{
      if (!event || !event.enabled) {{
        continue;
      }}
      if (!AudioCtx) {{
        continue;
      }}
      try {{
        const ctx = new AudioCtx();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        const frequencyMap = {{success: 880, failure: 220, warning: 440, running: 660}};
        osc.type = "sine";
        osc.frequency.value = frequencyMap[event.tone] || 550;
        gain.gain.value = Math.max(0.0, Math.min(1.0, event.volume || 0.35));
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.18);
      }} catch (error) {{
        console.debug("beep fallback", error);
      }}
    }}
    </script>
    """
    components.html(script, height=0)
