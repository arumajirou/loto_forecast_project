from __future__ import annotations

import pytest

from loto_forecast.api.streamlit import dashboard_db_admin_panel_validator as validator


def test_name_and_current_database_validation() -> None:
    assert validator.validate_non_empty_name("  sample  ", label="database") == "sample"
    with pytest.raises(ValueError):
        validator.validate_non_empty_name(" ", label="database")
    with pytest.raises(ValueError):
        validator.ensure_not_current_database("main", "main", action="drop")


def test_bulk_change_and_confirmation_helpers() -> None:
    validator.validate_bulk_change_allowed({"id": 1}, allow_all=False, action="update")
    with pytest.raises(ValueError):
        validator.validate_bulk_change_allowed({}, allow_all=False, action="delete")
    assert validator.expected_confirmation("DROP", "public.table") == "DROP public.table"
