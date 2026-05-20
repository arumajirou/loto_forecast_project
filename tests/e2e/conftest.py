from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "src/loto_forecast/api/streamlit/operations_dashboard.py"


def _playwright_available() -> bool:
    return importlib.util.find_spec("playwright.sync_api") is not None


def _should_run_e2e() -> bool:
    return os.getenv("RUN_STREAMLIT_E2E", "").strip() == "1" and _playwright_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _should_run_e2e():
        return
    skip = pytest.mark.skip(
        reason="Set RUN_STREAMLIT_E2E=1 and install playwright to run Streamlit browser E2E tests."
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_streamlit(base_url: str, timeout_sec: float = 60.0) -> None:
    deadline = time.time() + timeout_sec
    health_urls = [f"{base_url}/_stcore/health", base_url]
    while time.time() < deadline:
        for url in health_urls:
            try:
                with urllib.request.urlopen(url, timeout=2.0) as response:
                    if int(response.status) < 500:
                        return
            except urllib.error.URLError:
                pass
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit server did not become ready within {timeout_sec:.1f}s: {base_url}")


@pytest.fixture(scope="session")
def streamlit_base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not _should_run_e2e():
        pytest.skip("Playwright E2E is disabled.")

    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    mpl_dir = tmp_path_factory.mktemp("mplconfig")
    log_dir = PROJECT_ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"streamlit_e2e_server_{port}.log"
    log_fh = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    env.setdefault("SERVER_LOG_PATH", str(log_path))

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_streamlit(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_fh.close()


@pytest.fixture()
def browser_page(tmp_path: Path) -> Iterator[tuple[object, list[str], list[str], Path]]:
    if not _should_run_e2e():
        pytest.skip("Playwright E2E is disabled.")

    from playwright.sync_api import sync_playwright

    console_errors: list[str] = []
    page_errors: list[str] = []
    trace_path = tmp_path / "operations_dashboard_trace.zip"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1600, "height": 1200})
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        page.on(
            "console",
            lambda message: console_errors.append(f"{message.type}: {message.text}")
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        try:
            yield page, console_errors, page_errors, trace_path
        finally:
            context.tracing.stop(path=str(trace_path))
            context.close()
            browser.close()
