# V11 stuck-prevention patch

## Purpose

This patch prevents the screenshot workflow from appearing to hang at:

```bash
uv run --no-sync playwright install chromium
```

It also avoids recreating `.venv` while an existing Streamlit dashboard process is running.

## New recommended command

```bash
cd /mnt/e/env/fc/loto_forecast_project

export UV_LINK_MODE=copy
export LOTO_UV_ENV_MODE=browser
export LOTO_UV_CLEAR_VENV=0
export LOTO_PLAYWRIGHT_INSTALL=1
export LOTO_PLAYWRIGHT_INSTALL_TIMEOUT=900

./scripts/start_dashboard_and_capture_screenshots.sh --max-clicks 80 --max-depth 3
```

## Diagnostics

```bash
./scripts/diagnose_stuck_processes.sh
```

## Notes

- Set `LOTO_STOP_EXISTING_DASHBOARD=1` only when you intentionally want to restart the dashboard.
- Set `LOTO_UV_CLEAR_VENV=1` only when no dashboard process is running.
- If Playwright browser download is slow, increase `LOTO_PLAYWRIGHT_INSTALL_TIMEOUT`.
