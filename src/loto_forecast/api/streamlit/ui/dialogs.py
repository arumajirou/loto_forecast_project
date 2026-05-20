from __future__ import annotations

from typing import Any

import streamlit as st


def render_notification_dialog(event: dict[str, Any]) -> None:
    title = str(event.get("title", "通知"))
    severity = str(event.get("severity", "running"))
    tone = {
        "success": st.success,
        "failure": st.error,
        "warning": st.warning,
        "running": st.info,
    }.get(severity, st.info)
    with st.container(border=True):
        st.markdown(f"### {title}")
        tone(str(event.get("message", "")))
        next_actions = event.get("next_actions", [])
        if next_actions:
            st.caption("次にやること: " + " / ".join([str(item) for item in next_actions]))
