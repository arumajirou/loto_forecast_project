#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PORT="${LOTO_DASHBOARD_PORT:-8505}"
URL="${LOTO_DASHBOARD_URL:-http://localhost:${PORT}}"
LOG_DIR="${PROJECT_ROOT}/artifacts/observability/launcher"
LOG_FILE="${LOG_DIR}/streamlit_${PORT}.log"
PID_FILE="${LOG_DIR}/streamlit_${PORT}.pid"
READY_TIMEOUT="${LOTO_DASHBOARD_READY_TIMEOUT:-180}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ "${LOTO_SKIP_UV_SETUP:-0}" != "1" ]]; then
  # Reuse .venv by default. Recreating it here made the browser launcher slow and
  # could remove dependencies installed just before this script was called.
  LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}" ./scripts/setup_uv.sh
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    echo "Streamlit PID ${OLD_PID} already running; readiness will be checked."
  else
    rm -f "${PID_FILE}"
  fi
fi

if ! pgrep -f "streamlit run .*operations_dashboard.py.*${PORT}" >/dev/null 2>&1; then
  : > "${LOG_FILE}"
  nohup uv run --no-sync streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT}" \
    --server.headless true \
    --logger.level info \
    > "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "Started Streamlit PID $(cat "${PID_FILE}") on ${URL}"
fi

python - <<PY
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import urllib.request

url = "${URL}"
health_urls = [
    url.rstrip("/") + "/_stcore/health",
    url.rstrip("/") + "/healthz",
    url,
]
log_file = pathlib.Path("${LOG_FILE}")
pid_file = pathlib.Path("${PID_FILE}")
deadline = time.time() + int("${READY_TIMEOUT}")
last: object = None

def tail_log(n: int = 120) -> str:
    if not log_file.exists():
        return f"log file not found: {log_file}"
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:]) if lines else "(log file is empty)"

def pid_alive() -> bool:
    if not pid_file.exists():
        return True
    raw = pid_file.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return True
    try:
        os.kill(int(raw), 0)
        return True
    except OSError:
        return False
    except ValueError:
        return True

print(f"Waiting for dashboard readiness: {', '.join(health_urls)}", flush=True)
while time.time() < deadline:
    if not pid_alive():
        print("Streamlit process exited before readiness.", file=sys.stderr)
        print("----- streamlit log tail -----", file=sys.stderr)
        print(tail_log(), file=sys.stderr)
        raise SystemExit(1)
    for probe_url in health_urls:
        try:
            with urllib.request.urlopen(probe_url, timeout=3) as resp:
                if int(getattr(resp, "status", 200)) < 500:
                    print(f"ready: {probe_url}")
                    raise SystemExit(0)
        except Exception as exc:  # noqa: BLE001
            last = exc
    time.sleep(2)

print(f"not ready after ${READY_TIMEOUT}s: {url}: {last}", file=sys.stderr)
print("----- streamlit log tail -----", file=sys.stderr)
print(tail_log(), file=sys.stderr)
print("----- process check -----", file=sys.stderr)
subprocess.run(["bash", "-lc", "ps -eo pid,ppid,stat,etimes,cmd | grep -E 'streamlit|operations_dashboard' | grep -v grep || true"], check=False)
raise SystemExit(1)
PY

uv run --no-sync python scripts/collect_browser_observability.py --url "${URL}" "$@"
uv run --no-sync python scripts/observability_summary.py
