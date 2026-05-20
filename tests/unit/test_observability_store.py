from __future__ import annotations

from loto_forecast.observability.store import classify_event_level, detect_duplicate_events, stable_fingerprint


def test_classify_event_level_detects_traceback() -> None:
    assert classify_event_level("Traceback: ModuleNotFoundError") == "ERROR"


def test_duplicate_events_share_fingerprint() -> None:
    event = {
        "source": "test",
        "category": "browser",
        "level": "ERROR",
        "message": "same failure",
    }
    event["fingerprint"] = stable_fingerprint(event)
    duplicated = [dict(event), dict(event)]
    findings = detect_duplicate_events(duplicated)
    assert findings
    assert findings[0].count == 2
