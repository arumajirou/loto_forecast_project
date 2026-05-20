#!/usr/bin/env bash
set -euo pipefail

# Fast runner for unified dataset build.
# Optional overrides:
#   DB_HOST DB_PORT DB_USER DB_NAME (DB_PASSWORD is read from environment only)
#   BASE_SCHEMA BASE_TABLE HIST_SCHEMA HIST_TABLE EXOG_SCHEMA
#   OUTPUT_SCHEMA OUTPUT_TABLE
#   INCLUDE_EXOG_TABLES EXCLUDE_EXOG_TABLES
#   FAST_MODE(1/0) SHOW_PROGRESS(1/0) POSTGRES_CHUNKSIZE
#   POSTGRES_WRITE_MODE(to_sql/copy) POSTGRES_COPY_CHUNK_ROWS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DB_HOST="${DB_HOST:-${HOST:-127.0.0.1}}"
DB_PORT="${DB_PORT:-${PORT:-5432}}"
DB_USER="${DB_USER:-${LOTO_DB_USER:-loto}}"
DB_PASSWORD="${DB_PASSWORD:-${LOTO_DB_PASSWORD:-}}"
DB_NAME="${DB_NAME:-${DATABASE:-loto}}"

BASE_SCHEMA="${BASE_SCHEMA:-dataset}"
BASE_TABLE="${BASE_TABLE:-loto_y_ts}"
HIST_SCHEMA="${HIST_SCHEMA:-dataset}"
HIST_TABLE="${HIST_TABLE:-loto_hist_feat}"
EXOG_SCHEMA="${EXOG_SCHEMA:-exog}"

OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-dataset}"
OUTPUT_TABLE="${OUTPUT_TABLE:-loto_y_ts_unified}"

INCLUDE_EXOG_TABLES="${INCLUDE_EXOG_TABLES:-}"
EXCLUDE_EXOG_TABLES="${EXCLUDE_EXOG_TABLES:-}"
FAST_MODE="${FAST_MODE:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
POSTGRES_CHUNKSIZE="${POSTGRES_CHUNKSIZE:-20000}"
POSTGRES_WRITE_MODE="${POSTGRES_WRITE_MODE:-copy}"
POSTGRES_COPY_CHUNK_ROWS="${POSTGRES_COPY_CHUNK_ROWS:-20000}"

cmd=(
  python -m loto_forecast.cli build-unified-dataset
  --host "$DB_HOST"
  --port "$DB_PORT"
  --user "$DB_USER"
  --database "$DB_NAME"
  --base-schema "$BASE_SCHEMA"
  --base-table "$BASE_TABLE"
  --hist-schema "$HIST_SCHEMA"
  --hist-table "$HIST_TABLE"
  --exog-schema "$EXOG_SCHEMA"
  --output-schema "$OUTPUT_SCHEMA"
  --output-table "$OUTPUT_TABLE"
  --postgres-chunksize "$POSTGRES_CHUNKSIZE"
  --postgres-write-mode "$POSTGRES_WRITE_MODE"
  --postgres-copy-chunk-rows "$POSTGRES_COPY_CHUNK_ROWS"
)

if [[ "$FAST_MODE" == "1" ]]; then
  cmd+=(--fast-mode)
else
  cmd+=(--no-fast-mode)
fi

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  cmd+=(--show-progress)
else
  cmd+=(--no-show-progress)
fi

if [[ -n "$INCLUDE_EXOG_TABLES" ]]; then
  cmd+=(--include-exog-tables "$INCLUDE_EXOG_TABLES")
fi

if [[ -n "$EXCLUDE_EXOG_TABLES" ]]; then
  cmd+=(--exclude-exog-tables "$EXCLUDE_EXOG_TABLES")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[build_unified_dataset_fast] db=${DB_HOST}:${DB_PORT}/${DB_NAME} user=${DB_USER} base=${BASE_SCHEMA}.${BASE_TABLE} hist=${HIST_SCHEMA}.${HIST_TABLE} exog=${EXOG_SCHEMA} output=${OUTPUT_SCHEMA}.${OUTPUT_TABLE} fast_mode=${FAST_MODE}"
"${cmd[@]}"
