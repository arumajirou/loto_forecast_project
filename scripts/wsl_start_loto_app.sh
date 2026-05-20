#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${LOTO_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-dashboard}"
export LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}"

LOG_DIR="${PROJECT_ROOT}/artifacts/automation/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -d .venv ]]; then
  bash ./scripts/setup_uv.sh >>"${LOG_DIR}/setup_uv_dashboard.log" 2>&1
fi

if pgrep -f "streamlit run .*operations_dashboard.py" >/dev/null 2>&1; then
  echo "operations_dashboard.py is already running"
  exit 0
fi

nohup uv run --no-sync streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py \
  --server.address "${LOTO_DASHBOARD_HOST:-0.0.0.0}" \
  --server.port "${LOTO_DASHBOARD_PORT:-8505}" \
  >"${LOG_DIR}/dashboard_autostart.log" 2>&1 &

echo "$!" >"${LOG_DIR}/dashboard_autostart.pid"
echo "dashboard started pid=$(cat "${LOG_DIR}/dashboard_autostart.pid")"
