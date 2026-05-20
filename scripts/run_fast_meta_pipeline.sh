#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-loto}"
DB_PASSWORD="${DB_PASSWORD:-}"
DB_NAME="${DB_NAME:-loto}"
export DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME

BASE_SCHEMA="${BASE_SCHEMA:-dataset}"
BASE_TABLE="${BASE_TABLE:-loto_y_ts}"
HIST_SCHEMA="${HIST_SCHEMA:-dataset}"
HIST_TABLE="${HIST_TABLE:-loto_hist_feat}"
EXOG_SCHEMA="${EXOG_SCHEMA:-exog}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-dataset}"
OUTPUT_TABLE="${OUTPUT_TABLE:-loto_y_ts_unified}"
META_LIMIT="${META_LIMIT:-100}"

echo "[1/5] db-init"
python -m loto_forecast.cli db-init

echo "[2/5] meta-automodel-create"
bash "$ROOT_DIR/scripts/run_meta_automodel_create.sh"

echo "[3/5] build-unified-dataset (fast)"
python -m loto_forecast.cli build-unified-dataset \
  --host "$DB_HOST" --port "$DB_PORT" --user "$DB_USER" --database "$DB_NAME" \
  --base-schema "$BASE_SCHEMA" --base-table "$BASE_TABLE" \
  --hist-schema "$HIST_SCHEMA" --hist-table "$HIST_TABLE" \
  --exog-schema "$EXOG_SCHEMA" \
  --output-schema "$OUTPUT_SCHEMA" --output-table "$OUTPUT_TABLE" \
  --fast-mode --postgres-write-mode copy --postgres-copy-chunk-rows 50000 --show-progress

echo "[4/5] meta-automodel-run"
python -m loto_forecast.cli meta-automodel-run --limit "$META_LIMIT"

echo "[5/5] run-table-pyspark"
bash "$ROOT_DIR/scripts/run_table_pyspark.sh"

echo "done"
