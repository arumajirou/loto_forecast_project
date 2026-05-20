from __future__ import annotations

from typing import Any


def ensure_not_current_database(target_name: str, current_database: str, *, action: str) -> None:
    if target_name.strip() == current_database:
        raise ValueError(f"cannot {action} current connected database")


def validate_non_empty_name(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} is empty")
    return normalized


def validate_bulk_change_allowed(where_payload: dict[str, Any], *, allow_all: bool, action: str) -> None:
    if not where_payload and not allow_all:
        raise ValueError(f"where is empty. enable 'allow {action} all rows' to continue.")


def expected_confirmation(action: str, *parts: str) -> str:
    return " ".join([action, *[str(part).strip() for part in parts if str(part).strip()]])
