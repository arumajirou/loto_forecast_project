#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash ./scripts/capture_app_screenshots.sh --url http://localhost:8505 --max-clicks 80 --max-attempts 80 --max-depth 3

Environment:
  LOTO_UV_ENV_MODE=browser
  LOTO_PLAYWRIGHT_INSTALL=1  # optionally install chromium before capture
  LOTO_PLAYWRIGHT_INSTALL_TIMEOUT=300
EOF
  exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-browser}"
export LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}"

if [[ ! -d .venv ]]; then
  bash ./scripts/setup_uv.sh
fi

if [[ "${LOTO_PLAYWRIGHT_INSTALL:-0}" == "1" ]]; then
  echo "Installing Playwright chromium with timeout ${LOTO_PLAYWRIGHT_INSTALL_TIMEOUT:-300}s ..."
  if ! timeout "${LOTO_PLAYWRIGHT_INSTALL_TIMEOUT:-300}s" uv run --no-sync playwright install chromium; then
    cat >&2 <<'EOF'
Playwright chromium install did not finish.

Next options:
  1) Retry once with a longer timeout:
       LOTO_PLAYWRIGHT_INSTALL=1 LOTO_PLAYWRIGHT_INSTALL_TIMEOUT=900 bash ./scripts/capture_app_screenshots.sh --url http://localhost:8505
  2) Skip install if chromium is already installed:
       LOTO_PLAYWRIGHT_INSTALL=0 bash ./scripts/capture_app_screenshots.sh --url http://localhost:8505
  3) Check current processes:
       ./scripts/diagnose_stuck_processes.sh
EOF
    exit 124
  fi
fi

echo "Running browser observability collector ..."
echo "Progress format: [################--------------] processed/total percent | stage | message"
uv run --no-sync python scripts/collect_browser_observability.py "$@"
