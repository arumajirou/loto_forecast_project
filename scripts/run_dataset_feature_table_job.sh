#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ ! -d .venv ]]; then
  LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-static}" LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-1}" ./scripts/setup_uv.sh
fi

uv run --no-sync python scripts/run_dataset_feature_table_job.py "$@"
