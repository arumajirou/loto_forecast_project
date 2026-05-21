#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PAGER=cat
export PSQL_PAGER=cat

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
SAMPLE_LIMIT="${SAMPLE_LIMIT:-1000}"

if ! [[ "${TARGET_SCHEMA}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Invalid TARGET_SCHEMA: ${TARGET_SCHEMA}" >&2
  exit 2
fi

if ! [[ "${TARGET_TABLE}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Invalid TARGET_TABLE: ${TARGET_TABLE}" >&2
  exit 2
fi

if ! [[ "${SAMPLE_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "Invalid SAMPLE_LIMIT: ${SAMPLE_LIMIT}" >&2
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="artifacts/dataset_audit/${TS}"
mkdir -p "${OUT_DIR}"

echo "dataset audit package"
echo "target=${TARGET_SCHEMA}.${TARGET_TABLE}"
echo "sample_limit=${SAMPLE_LIMIT}"
echo "out_dir=${OUT_DIR}"

TARGET_EXISTS="$(
  psql -X -v ON_ERROR_STOP=1 -At -c "
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema='${TARGET_SCHEMA}'
        AND table_name='${TARGET_TABLE}'
    );
  "
)"

if [ "${TARGET_EXISTS}" != "t" ]; then
  echo "Target table does not exist: ${TARGET_SCHEMA}.${TARGET_TABLE}" >&2
  exit 3
fi

psql -X -v ON_ERROR_STOP=1 -c "SELECT current_database(), current_user, now();" \
  > "${OUT_DIR}/db_connection_check.txt"

psql -X -v ON_ERROR_STOP=1 -c "\dt *.*" \
  > "${OUT_DIR}/psql_tables.txt"

psql -X -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT table_schema, table_name, table_type
    FROM information_schema.tables
    WHERE table_schema IN ('dataset','exog','meta','model','resources','log','catalog')
    ORDER BY table_schema, table_name
  ) TO '${OUT_DIR}/tables_inventory.csv' CSV HEADER"

psql -X -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT table_schema, table_name, column_name, data_type, ordinal_position
    FROM information_schema.columns
    WHERE table_schema IN ('dataset','exog','meta','model','resources','log','catalog')
    ORDER BY table_schema, table_name, ordinal_position
  ) TO '${OUT_DIR}/columns_inventory.csv' CSV HEADER"

psql -X -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT column_name, data_type, ordinal_position
    FROM information_schema.columns
    WHERE table_schema = '${TARGET_SCHEMA}'
      AND table_name = '${TARGET_TABLE}'
    ORDER BY ordinal_position
  ) TO '${OUT_DIR}/target_columns.csv' CSV HEADER"

psql -X -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT
      '${TARGET_SCHEMA}' AS table_schema,
      '${TARGET_TABLE}' AS table_name,
      COUNT(*) AS row_count
    FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
  ) TO '${OUT_DIR}/target_row_count.csv' CSV HEADER"

HAS_DS="$(
  psql -X -v ON_ERROR_STOP=1 -At -c "
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema='${TARGET_SCHEMA}'
        AND table_name='${TARGET_TABLE}'
        AND column_name='ds'
    );
  "
)"

if [ "${HAS_DS}" = "t" ]; then
  psql -X -v ON_ERROR_STOP=1 \
    -c "\copy (
      SELECT
        COUNT(*) AS row_count,
        MIN(ds) AS min_ds,
        MAX(ds) AS max_ds
      FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    ) TO '${OUT_DIR}/target_time_range.csv' CSV HEADER"
else
  cat > "${OUT_DIR}/target_time_range.csv" <<CSV
row_count,min_ds,max_ds,note
,,,"column ds does not exist"
CSV
fi

HAS_UNIQUE_ID="$(
  psql -X -v ON_ERROR_STOP=1 -At -c "
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema='${TARGET_SCHEMA}'
        AND table_name='${TARGET_TABLE}'
        AND column_name='unique_id'
    );
  "
)"

if [ "${HAS_UNIQUE_ID}" = "t" ]; then
  psql -X -v ON_ERROR_STOP=1 \
    -c "\copy (
      SELECT COUNT(DISTINCT unique_id) AS unique_id_count
      FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    ) TO '${OUT_DIR}/target_unique_id_count.csv' CSV HEADER"
fi

HAS_CREATED="$(
  psql -X -v ON_ERROR_STOP=1 -At -c "
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema='${TARGET_SCHEMA}'
        AND table_name='${TARGET_TABLE}'
        AND column_name='created_at'
    );
  "
)"

HAS_UPDATED="$(
  psql -X -v ON_ERROR_STOP=1 -At -c "
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema='${TARGET_SCHEMA}'
        AND table_name='${TARGET_TABLE}'
        AND column_name='updated_at'
    );
  "
)"

if [ "${HAS_CREATED}" = "t" ] && [ "${HAS_UPDATED}" = "t" ]; then
  psql -X -v ON_ERROR_STOP=1 \
    -c "\copy (
      SELECT
        COUNT(*) AS row_count,
        MAX(created_at) AS max_created_at,
        MAX(updated_at) AS max_updated_at
      FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    ) TO '${OUT_DIR}/target_update_summary.csv' CSV HEADER"
elif [ "${HAS_CREATED}" = "t" ]; then
  psql -X -v ON_ERROR_STOP=1 \
    -c "\copy (
      SELECT
        COUNT(*) AS row_count,
        MAX(created_at) AS max_created_at,
        NULL::text AS max_updated_at
      FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    ) TO '${OUT_DIR}/target_update_summary.csv' CSV HEADER"
else
  psql -X -v ON_ERROR_STOP=1 \
    -c "\copy (
      SELECT
        COUNT(*) AS row_count,
        NULL::text AS max_created_at,
        NULL::text AS max_updated_at
      FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    ) TO '${OUT_DIR}/target_update_summary.csv' CSV HEADER"
fi

psql -X -v ON_ERROR_STOP=1 \
  -c "\copy (
    SELECT *
    FROM \"${TARGET_SCHEMA}\".\"${TARGET_TABLE}\"
    LIMIT ${SAMPLE_LIMIT}
  ) TO '${OUT_DIR}/target_sample_${SAMPLE_LIMIT}.csv' CSV HEADER"

psql -X -v ON_ERROR_STOP=1 \
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

psql -X -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM resources.run
    ORDER BY 1 DESC
    LIMIT 200
  ) TO '${OUT_DIR}/resources_run_recent.csv' CSV HEADER" \
  || true

psql -X -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM resources.stage_span
    ORDER BY 1 DESC
    LIMIT 500
  ) TO '${OUT_DIR}/resources_stage_span_recent.csv' CSV HEADER" \
  || true

psql -X -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM resources.resource_metric
    ORDER BY 1 DESC
    LIMIT 1000
  ) TO '${OUT_DIR}/resources_resource_metric_recent.csv' CSV HEADER" \
  || true

psql -X -v ON_ERROR_STOP=0 \
  -c "\copy (
    SELECT *
    FROM log.execution_event_log
    ORDER BY 1 DESC
    LIMIT 500
  ) TO '${OUT_DIR}/log_execution_event_recent.csv' CSV HEADER" \
  || true

psql -X -v ON_ERROR_STOP=0 \
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

Sample limit:

\`${SAMPLE_LIMIT}\`

Included files:

- db_connection_check.txt
- psql_tables.txt
- tables_inventory.csv
- columns_inventory.csv
- target_columns.csv
- target_row_count.csv
- target_time_range.csv
- target_unique_id_count.csv if unique_id exists
- target_update_summary.csv
- target_sample_${SAMPLE_LIMIT}.csv
- pg_stat_user_tables.csv
- resources_run_recent.csv if available
- resources_stage_span_recent.csv if available
- resources_resource_metric_recent.csv if available
- log_execution_event_recent.csv if available
- model_nf_automodel_recent.csv if available

Policy:

- This package uses SELECT only.
- dataset schema is treated as read-only.
- No db-init, write, delete, update, training, grid-run, or E2E is executed.
README

cat > "${OUT_DIR}/summary.json" <<JSON
{
  "generated_at_utc": "${TS}",
  "target_schema": "${TARGET_SCHEMA}",
  "target_table": "${TARGET_TABLE}",
  "sample_limit": ${SAMPLE_LIMIT},
  "read_only": true,
  "output_dir": "${OUT_DIR}"
}
JSON

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

echo "done"
