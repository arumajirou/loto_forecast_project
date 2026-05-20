#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed.

Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  exec "$SHELL" -l
EOF
  exit 127
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${LOTO_SKIP_UV_SETUP:-0}" != "1" ]]; then
  if ! uv run --no-sync python - <<'PY' >/dev/null 2>&1
import streamlit
import psycopg
PY
  then
    LOTO_UV_ENV_MODE=dashboard LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}" ./scripts/setup_uv.sh
  fi
fi

exec uv run --no-sync streamlit run "src/loto_forecast/api/streamlit/operations_dashboard.py" \
  --server.address "${LOTO_DASHBOARD_ADDRESS:-0.0.0.0}" \
  --server.port "${LOTO_DASHBOARD_PORT:-8505}" \
  --server.headless true \
  "$@"
