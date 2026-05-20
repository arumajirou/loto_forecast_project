#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PORT="${LOTO_DASHBOARD_PORT:-8505}"
URL="${LOTO_DASHBOARD_URL:-http://localhost:${PORT}}"
READY_TIMEOUT="${LOTO_DASHBOARD_READY_TIMEOUT:-180}"
INSTALL_TIMEOUT="${LOTO_PLAYWRIGHT_INSTALL_TIMEOUT:-300}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-browser}"
export LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage:
  ./scripts/start_dashboard_and_capture_screenshots.sh --max-clicks 80 --max-attempts 80 --max-depth 3

Safe defaults:
  - Reuses .venv unless LOTO_UV_CLEAR_VENV=1 is explicitly set.
  - Does not kill an existing dashboard unless LOTO_STOP_EXISTING_DASHBOARD=1 is set.
  - Playwright install is timeout-limited.
  - Invokes helper scripts via bash, so Windows/ZIP execute-bit loss does not block startup.

Environment:
  LOTO_DASHBOARD_PORT=8505
  LOTO_DASHBOARD_READY_TIMEOUT=180
  LOTO_PLAYWRIGHT_INSTALL=1
  LOTO_PLAYWRIGHT_INSTALL_TIMEOUT=300
  LOTO_STOP_EXISTING_DASHBOARD=0
EOF
  exit 0
fi

if [[ "${LOTO_STOP_EXISTING_DASHBOARD:-0}" == "1" ]]; then
  pkill -f "streamlit run .*operations_dashboard.py" 2>/dev/null || true
fi

if [[ ! -d .venv || "${LOTO_UV_CLEAR_VENV}" == "1" ]]; then
  bash ./scripts/setup_uv.sh
else
  echo "Reusing existing .venv. Set LOTO_UV_CLEAR_VENV=1 to recreate it."
fi

if [[ "${LOTO_PLAYWRIGHT_INSTALL:-1}" == "1" ]]; then
  echo "Checking Playwright chromium ..."
  if ! uv run --no-sync python scripts/check_playwright_chromium.py; then
    echo "Installing Playwright chromium with timeout ${INSTALL_TIMEOUT}s ..."
    timeout "${INSTALL_TIMEOUT}s" uv run --no-sync playwright install chromium
    uv run --no-sync python scripts/check_playwright_chromium.py
  fi
fi

# Start dashboard without clearing .venv.
LOTO_SKIP_UV_SETUP=1 bash ./scripts/wsl_start_loto_app.sh

python - <<PY
from __future__ import annotations

import pathlib
import sys
import time
import urllib.request

url = "${URL}"
deadline = time.time() + int("${READY_TIMEOUT}")
log_candidates = [
    pathlib.Path("artifacts/automation/logs/dashboard_autostart.log"),
    pathlib.Path("artifacts/observability/launcher/streamlit_${PORT}.log"),
]

def tail_logs() -> str:
    out = []
    for p in log_candidates:
        if p.exists():
            out.append(f"----- {p} -----")
            out.extend(p.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
    return "\\n".join(out) if out else "(no dashboard logs found)"

print(f"Waiting for dashboard: {url}", flush=True)
last = None
while time.time() < deadline:
    for probe in [url.rstrip("/") + "/_stcore/health", url]:
        try:
            with urllib.request.urlopen(probe, timeout=3) as resp:
                if int(getattr(resp, "status", 200)) < 500:
                    print(f"dashboard ready: {probe}")
                    raise SystemExit(0)
        except Exception as exc:  # noqa: BLE001
            last = exc
    time.sleep(2)

print(f"dashboard not ready: {url}: {last}", file=sys.stderr)
print(tail_logs(), file=sys.stderr)
raise SystemExit(1)
PY

echo "Starting browser screenshot capture with progress output ..."
echo "  url=${URL}"
echo "  args=$*"
bash ./scripts/capture_app_screenshots.sh --url "${URL}" "$@"
echo "Browser screenshot capture finished. Latest artifacts:"
find artifacts/observability/browser_runs -maxdepth 3 -type f 2>/dev/null | sort | tail -n 40 || true
