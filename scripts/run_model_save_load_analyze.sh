#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-loto}"
DB_PASSWORD="${DB_PASSWORD:-z}"
DB_NAME="${DB_NAME:-loto}"
export DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME

RUN_ID="${RUN_ID:-}"
AUTO_RESOLVE_RUN_ID="${AUTO_RESOLVE_RUN_ID:-true}"
CONFIG_NAME="${CONFIG_NAME:-}"
CONFIG_ID="${CONFIG_ID:-}"
RUN_STATUS="${RUN_STATUS:-success}"

resolve_latest_run_id() {
  local config_id="$1"
  local config_name="$2"
  local run_status="$3"
  python - "$config_id" "$config_name" "$run_status" <<'PY'
import sys
from sqlalchemy import text
from loto_forecast.config.settings import settings
from loto_forecast.data.db import make_engine

cfg_id_raw = str(sys.argv[1] or "").strip()
cfg_name = str(sys.argv[2] or "").strip() or None
run_status = str(sys.argv[3] or "").strip() or None
cfg_id = int(cfg_id_raw) if cfg_id_raw else None

def safe_ident(v: str) -> str:
    out = "".join(ch for ch in str(v) if ch.isalnum() or ch == "_")
    if not out:
        raise ValueError(f"invalid identifier: {v}")
    return out

model_schema = safe_ident(settings.model_schema)
model_table = safe_ident(settings.model_table)
meta_schema = safe_ident(settings.meta_schema)
meta_table = safe_ident(settings.meta_table)

q = text(
    f"""
    SELECT m.run_id
    FROM {model_schema}.{model_table} m
    LEFT JOIN {meta_schema}.{meta_table} c
      ON c.config_id = m.config_id
    WHERE m.run_id IS NOT NULL
      AND (:config_id IS NULL OR m.config_id = :config_id)
      AND (:config_name IS NULL OR c.config_name = :config_name)
      AND (:run_status IS NULL OR m.status = :run_status)
    ORDER BY m.ended_at DESC NULLS LAST, m.created_at DESC NULLS LAST, m.result_id DESC
    LIMIT 1
    """
)
engine = make_engine()
with engine.connect() as conn:
    row = conn.execute(
        q,
        {
            "config_id": cfg_id,
            "config_name": cfg_name,
            "run_status": run_status,
        },
    ).first()

if row is None:
    raise SystemExit(1)
print(str(row[0]))
PY
}

if [[ -z "$RUN_ID" ]]; then
  if [[ "$AUTO_RESOLVE_RUN_ID" == "true" ]]; then
    if RUN_ID="$(resolve_latest_run_id "$CONFIG_ID" "$CONFIG_NAME" "$RUN_STATUS")"; then
      echo "resolved RUN_ID=${RUN_ID} (config_name=${CONFIG_NAME:-<none>}, config_id=${CONFIG_ID:-<none>}, status=${RUN_STATUS})"
    else
      echo "RUN_ID resolution failed. set RUN_ID explicitly or adjust CONFIG_NAME/CONFIG_ID/RUN_STATUS." >&2
      exit 1
    fi
  else
    echo "RUN_ID is required when AUTO_RESOLVE_RUN_ID=false." >&2
    exit 1
  fi
fi

SOURCE_PATH="${SOURCE_PATH:-}"
SAVE_PATH="${SAVE_PATH:-./artifacts/saved_models/{run_id}}"
RUN_SAVE="${RUN_SAVE:-true}"
RUN_LOAD="${RUN_LOAD:-true}"
RUN_ANALYZE="${RUN_ANALYZE:-true}"
SAVE_DATASET="${SAVE_DATASET:-false}"
SAVE_OVERWRITE="${SAVE_OVERWRITE:-true}"
LOAD_CHECK_PREDICT="${LOAD_CHECK_PREDICT:-false}"
INSAMPLE_STEP_SIZE="${INSAMPLE_STEP_SIZE:-1}"

if [[ "$SAVE_PATH" == *"{run_id}"* ]]; then
  SAVE_PATH="${SAVE_PATH//\{run_id\}/$RUN_ID}"
fi
if [[ -n "$SOURCE_PATH" && "$SOURCE_PATH" == *"{run_id}"* ]]; then
  SOURCE_PATH="${SOURCE_PATH//\{run_id\}/$RUN_ID}"
fi

CMD=(
  python -m loto_forecast.cli model-save-load-analyze
  --run-id "$RUN_ID"
  --save-path "$SAVE_PATH"
  --insample-step-size "$INSAMPLE_STEP_SIZE"
)

if [[ -n "$SOURCE_PATH" ]]; then
  CMD+=(--source-path "$SOURCE_PATH")
fi
if [[ "$RUN_SAVE" == "true" ]]; then
  CMD+=(--run-save)
else
  CMD+=(--no-run-save)
fi
if [[ "$RUN_LOAD" == "true" ]]; then
  CMD+=(--run-load)
else
  CMD+=(--no-run-load)
fi
if [[ "$RUN_ANALYZE" == "true" ]]; then
  CMD+=(--run-analyze)
else
  CMD+=(--no-run-analyze)
fi
if [[ "$SAVE_DATASET" == "true" ]]; then
  CMD+=(--save-dataset)
else
  CMD+=(--no-save-dataset)
fi
if [[ "$SAVE_OVERWRITE" == "true" ]]; then
  CMD+=(--save-overwrite)
else
  CMD+=(--no-save-overwrite)
fi
if [[ "$LOAD_CHECK_PREDICT" == "true" ]]; then
  CMD+=(--load-check-predict)
else
  CMD+=(--no-load-check-predict)
fi

"${CMD[@]}"
