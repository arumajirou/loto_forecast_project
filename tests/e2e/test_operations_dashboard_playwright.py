from __future__ import annotations

from pathlib import Path

import pytest

from .pages.operations_dashboard_page import OperationsDashboardPage

pytestmark = [pytest.mark.e2e]


def test_operations_dashboard_shell_and_train_flow(
    streamlit_base_url: str,
    browser_page: tuple[object, list[str], list[str], Path],
    tmp_path: Path,
) -> None:
    page, console_errors, page_errors, trace_path = browser_page
    dashboard = OperationsDashboardPage(page=page, base_url=streamlit_base_url)

    dashboard.goto()
    dashboard.expect_shell_loaded()
    dashboard.expect_db_feedback()
    dashboard.save_screenshot(tmp_path / "operations_dashboard_home.png")

    dashboard.open_nf_lab_train()
    dashboard.save_screenshot(tmp_path / "operations_dashboard_nf_train.png")

    snapshot = dashboard.accessibility_snapshot()
    assert snapshot is not None
    assert not console_errors
    assert not page_errors
    assert trace_path.suffix == ".zip"


def test_operations_dashboard_operations_fallback_tabs(
    streamlit_base_url: str,
    browser_page: tuple[object, list[str], list[str], Path],
    tmp_path: Path,
) -> None:
    page, console_errors, page_errors, trace_path = browser_page
    dashboard = OperationsDashboardPage(page=page, base_url=streamlit_base_url)

    dashboard.goto()
    dashboard.open_operations_fallback()
    dashboard.save_screenshot(tmp_path / "operations_dashboard_operations_fallback.png")

    assert not console_errors
    assert not page_errors
    assert trace_path.suffix == ".zip"
