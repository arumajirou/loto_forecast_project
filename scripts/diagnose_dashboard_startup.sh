#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PORT="${LOTO_DASHBOARD_PORT:-8505}"
URL="${LOTO_DASHBOARD_URL:-http://localhost:${PORT}}"
LOG_DIR="${PROJECT_ROOT}/artifacts/observability/launcher"
LOG_FILE="${LOG_DIR}/streamlit_${PORT}.log"
PID_FILE="${LOG_DIR}/streamlit_${PORT}.pid"

echo "== dashboard diagnose =="
echo "project: ${PROJECT_ROOT}"
echo "url: ${URL}"
echo "log: ${LOG_FILE}"
echo "pid file: ${PID_FILE}"
echo

echo "== process =="
ps -eo pid,ppid,stat,etimes,cmd | grep -E 'streamlit|operations_dashboard' | grep -v grep || true
echo

echo "== ports =="
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) | grep -E ":${PORT}\b|State|Proto" || true
echo

echo "== health probes =="
python - <<PY
import urllib.request
for url in ["${URL.rstrip('/')}/_stcore/health", "${URL.rstrip('/')}/healthz", "${URL}"]:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            print(url, "=>", getattr(resp, "status", "unknown"))
    except Exception as exc:
        print(url, "=>", type(exc).__name__, exc)
PY
echo

echo "== recent launcher log =="
if [[ -f "${LOG_FILE}" ]]; then
  tail -n 200 "${LOG_FILE}"
else
  echo "log file not found"
fi
