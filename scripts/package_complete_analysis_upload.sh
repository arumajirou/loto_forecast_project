#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-sync python scripts/package_analysis_upload.py "$@"
fi
exec "${PYTHON_BIN}" scripts/package_analysis_upload.py "$@"
