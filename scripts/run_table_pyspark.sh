#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DB_HOST="${DB_HOST:-127.0.0.1}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-loto}"
DB_PASSWORD="${DB_PASSWORD:-}"
DB_NAME="${DB_NAME:-loto}"

SOURCE_SCHEMA="${SOURCE_SCHEMA:-dataset}"
SOURCE_TABLE="${SOURCE_TABLE:-loto_y_ts_unified}"
TARGET_SCHEMA="${TARGET_SCHEMA:-dataset}"
TARGET_TABLE="${TARGET_TABLE:-loto_y_ts_unified_spark}"
if [[ -z "${SOURCE_SQL:-}" ]]; then
  SOURCE_SQL="SELECT * FROM \"${SOURCE_SCHEMA}\".\"${SOURCE_TABLE}\" WHERE y IS NOT NULL"
fi
TRANSFORM_SQL="${TRANSFORM_SQL:-}"
OUTPUT_IF_EXISTS="${OUTPUT_IF_EXISTS:-replace}"
OUTPUT_PARQUET_PATH="${OUTPUT_PARQUET_PATH:-./artifacts/datasets/loto_y_ts_unified_spark.parquet}"
REPARTITION="${REPARTITION:-0}"
SPARK_MASTER="${SPARK_MASTER:-}"
EXECUTION_BACKEND="${EXECUTION_BACKEND:-auto}"
DASK_NPARTITIONS="${DASK_NPARTITIONS:-0}"
PREFER_PANDAS="${PREFER_PANDAS:-false}"
SKIP_ROW_COUNT="${SKIP_ROW_COUNT:-true}"
SPARK_UI_ENABLED="${SPARK_UI_ENABLED:-false}"
SPARK_SHUFFLE_PARTITIONS="${SPARK_SHUFFLE_PARTITIONS:-16}"
SPARK_READER_FETCHSIZE="${SPARK_READER_FETCHSIZE:-10000}"
POSTGRES_WRITE_MODE="${POSTGRES_WRITE_MODE:-copy}"
POSTGRES_COPY_CHUNK_ROWS="${POSTGRES_COPY_CHUNK_ROWS:-50000}"
POSTGRES_LOCK_TIMEOUT_MS="${POSTGRES_LOCK_TIMEOUT_MS:-10000}"

CMD=(
  python -m loto_forecast.cli run-table-pyspark
  --host "$DB_HOST" --port "$DB_PORT" --user "$DB_USER" --database "$DB_NAME"
  --source-schema "$SOURCE_SCHEMA" --source-table "$SOURCE_TABLE"
  --source-sql "$SOURCE_SQL"
  --target-schema "$TARGET_SCHEMA" --target-table "$TARGET_TABLE"
  --output-if-exists "$OUTPUT_IF_EXISTS"
  --output-parquet-path "$OUTPUT_PARQUET_PATH"
  --execution-backend "$EXECUTION_BACKEND"
)
if [[ "$DASK_NPARTITIONS" != "0" ]]; then CMD+=(--dask-npartitions "$DASK_NPARTITIONS"); fi
if [[ -n "$TRANSFORM_SQL" ]]; then
  CMD+=(--transform-sql "$TRANSFORM_SQL")
fi

if [[ "$REPARTITION" != "0" ]]; then
  CMD+=(--repartition "$REPARTITION")
fi
if [[ -n "$SPARK_MASTER" ]]; then
  CMD+=(--spark-master "$SPARK_MASTER")
fi
if [[ "$PREFER_PANDAS" == "true" ]]; then CMD+=(--prefer-pandas); else CMD+=(--no-prefer-pandas); fi
if [[ "$SKIP_ROW_COUNT" == "true" ]]; then CMD+=(--skip-row-count); else CMD+=(--no-skip-row-count); fi
if [[ "$SPARK_UI_ENABLED" == "true" ]]; then CMD+=(--spark-ui-enabled); else CMD+=(--no-spark-ui-enabled); fi
if [[ "${SPARK_SHUFFLE_PARTITIONS}" != "" ]]; then CMD+=(--spark-shuffle-partitions "$SPARK_SHUFFLE_PARTITIONS"); fi
if [[ "${SPARK_READER_FETCHSIZE}" != "" ]]; then CMD+=(--spark-reader-fetchsize "$SPARK_READER_FETCHSIZE"); fi
CMD+=(--postgres-write-mode "$POSTGRES_WRITE_MODE")
if [[ "${POSTGRES_COPY_CHUNK_ROWS}" != "" ]]; then CMD+=(--postgres-copy-chunk-rows "$POSTGRES_COPY_CHUNK_ROWS"); fi
if [[ "${POSTGRES_LOCK_TIMEOUT_MS}" != "" ]]; then CMD+=(--postgres-lock-timeout-ms "$POSTGRES_LOCK_TIMEOUT_MS"); fi

"${CMD[@]}"
