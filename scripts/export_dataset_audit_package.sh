#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f ".env" ]; then
  set -a
  source ".env"
  set +a
fi

export PGHOST="${DB_HOST:-127.0.0.1}"
export PGPORT="${DB_PORT:-5432}"
export PGUSER="${DB_USER:-loto}"
export PGDATABASE="${DB_NAME:-loto}"
export PGPASSWORD="${DB_PASSWORD:-}"

TARGET_SCHEMA="${TARGET_SCHEMA:-dataset}"
TARGET_TABLE="${TARGET_TABLE:-loto_y_ts_unified}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="artifacts/dataset_audit/${TS}"
mkdir -p "${OUT_DIR}"

echo "dataset audit package"
echo "target=${TARGET_SCHEMA}.${TARGET_TABLE}"
echo "out_dir=${OUT_DIR}"

psql -v ON_ERROR_STOP=1 -c "SELECT current_database(), current_user, now();" \
  > "${OUT_DIR}/db_connection_check.txt"

psql -v ON_ERROR_STOP=1 -c "\dt *.*" \
  > "${OUT_DIR}/psql_tables.txt"

psql -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT table_schema, table_name, table_type
    FROM information_schema.tables
    WHERE table_schema IN ('dataset','exog','meta','model','resources','log','catalog')
    ORDER BY table_schema, table_name
  ) TO '${OUT_DIR}/tables_inventory.csv' CSV HEADER"

psql -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT table_schema, table_name, column_name, data_type, ordinal_position
    FROM information_schema.columns
    WHERE table_schema IN ('dataset','exog','meta','model','resources','log','catalog')
    ORDER BY table_schema, table_name, ordinal_position
  ) TO '${OUT_DIR}/columns_inventory.csv' CSV HEADER"

psql -v ON_ERROR_STOP=1 \
  -v schema="${TARGET_SCHEMA}" \
  -v table="${TARGET_TABLE}" \
  -c "\copy (
    SELECT column_name, data_type, ordinal_position
    FROM information_schema.columns
    WHERE table_schema = :'schema'
      AND table_name = :'table'
    ORDER BY ordinal_position
  ) TO '${OUT_DIR}/target_columns.csv' CSV HEADER"

psql -v ON_ERROR_STOP=1 \
  -v schema="${TARGET_SCHEMA}" \
  -v table="${TARGET_TABLE}" \
  -c "\copy (
    SELECT
      :'schema' AS table_schema,
      :'table' AS table_name,
      COUNT(*) AS row_count
    FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
  ) TO '${OUT_DIR}/target_row_count.csv' CSV HEADER"

psql -v ON_ERROR_STOP=1 \
  -v schema="${TARGET_SCHEMA}" \
  -v table="${TARGET_TABLE}" \
  -c "\copy (
    SELECT *
    FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    LIMIT 1000
  ) TO '${OUT_DIR}/target_sample_1000.csv' CSV HEADER"

psql -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT
      schemaname,
      relname AS table_name,
      n_live_tup,
      n_dead_tup,
      last_vacuum,
      last_autovacuum,
      last_analyze,
      last_autoanalyze
    FROM pg_stat_user_tables
    WHERE schemaname IN ('dataset','exog','meta','model','resources','log','catalog')
    ORDER BY schemaname, relname
  ) TO '${OUT_DIR}/pg_stat_user_tables.csv' CSV HEADER"

# 実行ログ系テーブルがある場合だけ出力
psql -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM resources.run
    ORDER BY 1 DESC
    LIMIT 200
  ) TO '${OUT_DIR}/resources_run_recent.csv' CSV HEADER" \
  || true

psql -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM resources.stage_span
    ORDER BY 1 DESC
    LIMIT 500
  ) TO '${OUT_DIR}/resources_stage_span_recent.csv' CSV HEADER" \
  || true

psql -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM model.nf_automodel
    ORDER BY 1 DESC
    LIMIT 200
  ) TO '${OUT_DIR}/model_nf_automodel_recent.csv' CSV HEADER" \
  || true

cat > "${OUT_DIR}/README.md" <<README
# Dataset Audit Package

Generated at: ${TS}

Target table:

\`${TARGET_SCHEMA}.${TARGET_TABLE}\`

Included files:

- db_connection_check.txt
- psql_tables.txt
- tables_inventory.csv
- columns_inventory.csv
- target_columns.csv
- target_row_count.csv
- target_sample_1000.csv
- pg_stat_user_tables.csv
- resources_run_recent.csv if available
- resources_stage_span_recent.csv if available
- model_nf_automodel_recent.csv if available

Policy:

- This package uses SELECT only.
- dataset schema is treated as read-only.
- No db-init, write, delete, update, training, or grid-run is executed.
README

ZIP_PATH="artifacts/dataset_audit/dataset_audit_${TARGET_SCHEMA}_${TARGET_TABLE}_${TS}.zip"
LATEST_PATH="artifacts/dataset_audit/latest_dataset_audit_package.zip"

python - <<PY
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
out_dir = Path("${OUT_DIR}")
zip_path = Path("${ZIP_PATH}")
latest_path = Path("${LATEST_PATH}")
with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(out_dir.parent))
latest_path.write_bytes(zip_path.read_bytes())
print(zip_path)
print(latest_path)
PY
