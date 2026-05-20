#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed.

Install uv first, then rerun this script:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  exec "$SHELL" -l
EOF
  exit 127
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-static}"
LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}"

export UV_LINK_MODE

echo "Project root: ${PROJECT_ROOT}"
echo "Python version request: ${PYTHON_VERSION}"
echo "UV link mode: ${UV_LINK_MODE}"
echo "Environment mode: ${LOTO_UV_ENV_MODE}"
echo "Clear existing .venv: ${LOTO_UV_CLEAR_VENV}"
echo "Tip: set LOTO_UV_CLEAR_VENV=1 only when no dashboard process is running."

# Remove stale locks generated in other environments. A lock file created behind
# a private package mirror can pin direct archive URLs that are unreachable on
# this machine, and a CUDA lock can force nvidia/cuda wheels in CPU-only checks.
if [[ "${LOTO_KEEP_UV_LOCK:-0}" != "1" ]]; then
  rm -f uv.lock
fi

if [[ "${LOTO_UV_CLEAR_VENV}" == "1" ]]; then
  rm -rf .venv
  uv venv --python "${PYTHON_VERSION}" --clear
else
  if [[ -d .venv ]]; then
    echo "Reusing existing .venv because LOTO_UV_CLEAR_VENV=0"
  else
    uv venv --python "${PYTHON_VERSION}"
  fi
fi

case "${LOTO_UV_ENV_MODE}" in
  static)
    uv sync --extra dev
    ;;
  dashboard)
    uv sync --extra dev --extra dashboard
    ;;
  browser)
    uv sync --extra dev --extra dashboard --extra browser
    ;;
  observability)
    uv sync --extra dev --extra observability
    ;;
  full)
    uv sync --extra dev --extra full
    ;;
  *)
    echo "Unknown LOTO_UV_ENV_MODE='${LOTO_UV_ENV_MODE}'. Use static, dashboard, browser, observability, or full." >&2
    exit 2
    ;;
esac

echo
echo "uv environment is ready."
echo "Examples:"
echo "  ./scripts/verify_static.sh"
echo "  ./scripts/run_dashboard_observability.sh --max-clicks 20"
echo "  PYTHONPATH=src uv run --no-sync python -m compileall -q src tests tools evals scripts"
