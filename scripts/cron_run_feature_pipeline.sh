#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${LOTO_PROJECT_ROOT:-/mnt/e/env/fc/loto_forecast_project}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-static}"
export LOTO_UV_CLEAR_VENV="${LOTO_UV_CLEAR_VENV:-0}"

LOG_DIR="${PROJECT_ROOT}/artifacts/automation/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -d .venv ]]; then
  ./scripts/setup_uv.sh >>"${LOG_DIR}/setup_uv.log" 2>&1
fi

ARGS=(
  --source-schema "${LOTO_FEATURE_SOURCE_SCHEMA:-dataset}"
  --source-table "${LOTO_FEATURE_SOURCE_TABLE:-loto_y_ts_unified}"
  --target-schema "${LOTO_FEATURE_TARGET_SCHEMA:-exog}"
  --target-table "${LOTO_FEATURE_TARGET_TABLE:-nf_feature_table_auto}"
  --limit "${LOTO_FEATURE_LIMIT:-5000}"
)

if [[ "${LOTO_CRON_FEATURE_WRITE:-0}" == "1" ]]; then
  export LOTO_ALLOW_FEATURE_DB_WRITE="${LOTO_ALLOW_FEATURE_DB_WRITE:-1}"
  ARGS+=(--yes-write)
fi

ts="$(date +%Y%m%d_%H%M%S)"
./scripts/run_dataset_feature_table_job.sh "${ARGS[@]}" >>"${LOG_DIR}/feature_job_${ts}.log" 2>&1
