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

RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1800}"
PROGRESS_HEARTBEAT_SECONDS="${PROGRESS_HEARTBEAT_SECONDS:-10}"
CONSOLE_LOG_MODE="${CONSOLE_LOG_MODE:-all}" # all | progress
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/pipeline_runs}"
mkdir -p "$LOG_DIR"
ONE_MODEL_ONLY="${ONE_MODEL_ONLY:-false}"

CONFIG_NAME="${CONFIG_NAME:-local_nf_run_01}"
BASE_SCHEMA="${BASE_SCHEMA:-dataset}"
BASE_TABLE="${BASE_TABLE:-loto_y_ts_unified_spark}"
HIST_SCHEMA="${HIST_SCHEMA:-dataset}"
HIST_TABLE="${HIST_TABLE:-loto_hist_feat}"
EXOG_SCHEMA="${EXOG_SCHEMA:-exog}"
OUTPUT_SCHEMA="${OUTPUT_SCHEMA:-dataset}"
OUTPUT_TABLE="${OUTPUT_TABLE:-loto_y_ts_unified}"
MODEL_NAME="${MODEL_NAME:-AutoNHITS}"
HORIZON="${HORIZON:-28}"
TARGET_LOTO="${TARGET_LOTO:-bingo5}"
TARGET_UNIQUE_ID="${TARGET_UNIQUE_ID:-N1}"
TARGET_TS_TYPE="${TARGET_TS_TYPE:-raw}"

AUTO_CLS_MODEL="${AUTO_CLS_MODEL:-AutoNHITS}"
AUTO_H="${AUTO_H:-28}"
AUTO_LOSS="${AUTO_LOSS:-MAE}"
AUTO_VALID_LOSS="${AUTO_VALID_LOSS:-MAE}"
AUTO_SEARCH_ALG="${AUTO_SEARCH_ALG:-BasicVariantGenerator}"
AUTO_NUM_SAMPLES="${AUTO_NUM_SAMPLES:-10}"
AUTO_CPUS="${AUTO_CPUS:-0}"
AUTO_GPUS="${AUTO_GPUS:-0}"
AUTO_VERBOSE="${AUTO_VERBOSE:-true}"
AUTO_BACKEND="${AUTO_BACKEND:-optuna}"
RECURSIVE_DEPTH="${RECURSIVE_DEPTH:-2}"
MAX_TASKS="${MAX_TASKS:-0}"

RUN_PREDICT="${RUN_PREDICT:-true}"
RUN_EVALUATE="${RUN_EVALUATE:-true}"
RUN_EXPLAIN_IS_SET="${RUN_EXPLAIN+x}"
RUN_EXPLAIN="${RUN_EXPLAIN:-true}"
RUN_SAVE="${RUN_SAVE:-true}"
RUN_LOAD="${RUN_LOAD:-true}"
RUN_ANALYZE="${RUN_ANALYZE:-true}"
SAVE_DATASET="${SAVE_DATASET:-false}"
SAVE_OVERWRITE="${SAVE_OVERWRITE:-true}"
LOAD_CHECK_PREDICT="${LOAD_CHECK_PREDICT:-false}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/artifacts/saved_models/{run_id}}"

if [[ -z "${AUTO_CONFIG_JSON:-}" ]]; then
  AUTO_CONFIG_JSON='{"backend":"optuna","num_samples":10}'
fi
if [[ -z "${UNIFIED_FILTER_JSON:-}" ]]; then
  UNIFIED_FILTER_JSON="{\"loto\":\"${TARGET_LOTO}\",\"unique_id\":\"${TARGET_UNIQUE_ID}\",\"ts_type\":\"${TARGET_TS_TYPE}\"}"
fi
if [[ -z "${UNIFIED_GROUP_COLS_JSON:-}" ]]; then
  UNIFIED_GROUP_COLS_JSON='["loto","unique_id","ts_type"]'
fi
UNIFIED_GROUP_VALIDATE_STRICT="${UNIFIED_GROUP_VALIDATE_STRICT:-false}"
if [[ -z "${MODEL_PARAMS_JSON:-}" ]]; then
  MODEL_PARAMS_JSON='{"backend":"optuna","num_samples":20}'
fi
if [[ -z "${PARAM_SPACE_JSON:-}" ]]; then
  PARAM_SPACE_JSON='{"num_samples":[10,20],"seed":[1,2]}'
fi
if [[ -z "${AUTO_CALLBACKS_JSON:-}" ]]; then
  AUTO_CALLBACKS_JSON='[]'
fi

if [[ "$CONSOLE_LOG_MODE" != "all" && "$CONSOLE_LOG_MODE" != "progress" ]]; then
  echo "invalid CONSOLE_LOG_MODE=${CONSOLE_LOG_MODE}. use all|progress"
  exit 1
fi

if [[ "$ONE_MODEL_ONLY" == "true" ]]; then
  RECURSIVE_DEPTH=1
  MAX_TASKS=1
  if [[ -z "$RUN_EXPLAIN_IS_SET" ]]; then
    RUN_EXPLAIN=false
  fi
fi

SOURCE_SCHEMA="${SOURCE_SCHEMA:-dataset}"
SOURCE_TABLE="${SOURCE_TABLE:-loto_y_ts_unified}"
TARGET_SCHEMA="${TARGET_SCHEMA:-dataset}"
TARGET_TABLE="${TARGET_TABLE:-loto_y_ts_unified_spark}"
if [[ -z "${SOURCE_SQL:-}" ]]; then
  SOURCE_SQL="SELECT * FROM \"${SOURCE_SCHEMA}\".\"${SOURCE_TABLE}\" WHERE y IS NOT NULL AND loto = '${TARGET_LOTO}' AND unique_id = '${TARGET_UNIQUE_ID}' AND ts_type = '${TARGET_TS_TYPE}'"
fi
TRANSFORM_SQL="${TRANSFORM_SQL:-}"
OUTPUT_IF_EXISTS="${OUTPUT_IF_EXISTS:-replace}"
OUTPUT_PARQUET_PATH="${OUTPUT_PARQUET_PATH:-$ROOT_DIR/artifacts/datasets/loto_y_ts_unified_spark.parquet}"
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

RUN_CHECK_GROUPING="${RUN_CHECK_GROUPING:-true}"
CHECK_SCHEMA="${CHECK_SCHEMA:-$SOURCE_SCHEMA}"
CHECK_TABLE="${CHECK_TABLE:-$TARGET_TABLE}"
CHECK_GROUP_COLS="${CHECK_GROUP_COLS:-loto,unique_id,ts_type}"
CHECK_TIME_COL="${CHECK_TIME_COL:-ds}"
RUN_META_AUTOMODEL_RUN="${RUN_META_AUTOMODEL_RUN:-true}"
META_RUN_CONFIG_ID="${META_RUN_CONFIG_ID:-}"
META_LIMIT="${META_LIMIT:-1}"
META_STOP_ON_ERROR="${META_STOP_ON_ERROR:-true}"

RUN_TAG="$(date '+%Y%m%d_%H%M%S')"

_slug() {
  echo "$1" | tr ' /:' '___'
}

_render_progress_bar() {
  local tick="$1"
  local width=24
  local pos=$((tick % width))
  local left right
  left="$(printf '%*s' "$pos" '' | tr ' ' '-')"
  right="$(printf '%*s' "$((width - pos - 1))" '' | tr ' ' '-')"
  printf "%s>%s" "$left" "$right"
}

resolve_config_id_by_name() {
  local config_name="$1"
  python - "$config_name" <<'PY'
import sys
from sqlalchemy import text
from loto_forecast.config.settings import settings
from loto_forecast.data.db import make_engine

config_name = sys.argv[1]
schema = "".join(ch for ch in str(settings.meta_schema) if ch.isalnum() or ch == "_")
table = "".join(ch for ch in str(settings.meta_table) if ch.isalnum() or ch == "_")
if not schema or not table:
    raise ValueError("invalid meta schema/table")

q = text(
    f"""
    SELECT config_id
    FROM {schema}.{table}
    WHERE config_name = :config_name
    ORDER BY config_id DESC
    LIMIT 1
    """
)
engine = make_engine()
with engine.connect() as conn:
    row = conn.execute(q, {"config_name": config_name}).first()

if row is None:
    raise SystemExit(1)
print(int(row[0]))
PY
}

run_cmd() {
  local step="$1"
  local total="$2"
  local label="$3"
  shift 3
  local started ended elapsed rc
  local log_file="$LOG_DIR/${RUN_TAG}_$(printf '%02d' "$step")_$(_slug "$label").log"
  local cmd_pid heart_s line_count last_line

  started=$(date +%s)
  echo "[$step/$total] $(date '+%Y-%m-%d %H:%M:%S') START ${label}"
  echo "[$step/$total] log: $log_file"
  : > "$log_file"
  set +e
  if [[ "${RUN_TIMEOUT_SECONDS}" -gt 0 ]]; then
    if [[ "$CONSOLE_LOG_MODE" == "progress" ]]; then
      timeout --foreground "${RUN_TIMEOUT_SECONDS}s" "$@" >>"$log_file" 2>&1 &
    else
      timeout --foreground "${RUN_TIMEOUT_SECONDS}s" "$@" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2) &
    fi
  else
    if [[ "$CONSOLE_LOG_MODE" == "progress" ]]; then
      "$@" >>"$log_file" 2>&1 &
    else
      "$@" > >(tee -a "$log_file") 2> >(tee -a "$log_file" >&2) &
    fi
  fi
  cmd_pid=$!
  if [[ "${PROGRESS_HEARTBEAT_SECONDS}" =~ ^[0-9]+$ ]] && [[ "${PROGRESS_HEARTBEAT_SECONDS}" -gt 0 ]]; then
    heart_s="${PROGRESS_HEARTBEAT_SECONDS}"
  else
    heart_s=10
  fi
  while kill -0 "$cmd_pid" 2>/dev/null; do
    sleep "$heart_s"
    if ! kill -0 "$cmd_pid" 2>/dev/null; then
      break
    fi
    elapsed=$(( $(date +%s) - started ))
    line_count="$(wc -l < "$log_file" 2>/dev/null || echo 0)"
    last_line="$(tail -n 1 "$log_file" 2>/dev/null | tr '\r' '\n' | tail -n 1 | sed -r 's/\x1B\[[0-9;?]*[[:alpha:]]//g' | cut -c1-180)"
    if [[ "$CONSOLE_LOG_MODE" == "progress" ]]; then
      local tick bar
      tick=$((elapsed / heart_s))
      bar="$(_render_progress_bar "$tick")"
      printf "\r[%s/%s] RUNNING %-24s [%s] elapsed=%ss log_lines=%s" \
        "$step" "$total" "$label" "$bar" "$elapsed" "$line_count"
    else
      echo "[$step/$total] $(date '+%Y-%m-%d %H:%M:%S') RUNNING ${label} elapsed=${elapsed}s log_lines=${line_count} last='${last_line}'"
    fi
  done
  if [[ "$CONSOLE_LOG_MODE" == "progress" ]]; then
    printf "\n"
  fi
  wait "$cmd_pid"
  rc=$?
  set -e
  ended=$(date +%s)
  elapsed=$((ended - started))
  if [[ $rc -eq 0 ]]; then
    echo "[$step/$total] $(date '+%Y-%m-%d %H:%M:%S') DONE ${label} elapsed=${elapsed}s"
  else
    echo "[$step/$total] $(date '+%Y-%m-%d %H:%M:%S') FAIL ${label} rc=${rc} elapsed=${elapsed}s"
    exit "$rc"
  fi
}

run_step() {
  local step="$1"
  local total="$2"
  local label="$3"
  echo "[$step/$total] $(date '+%Y-%m-%d %H:%M:%S') ${label}"
}

TOTAL_STEPS=2
if [[ "$RUN_CHECK_GROUPING" == "true" ]]; then TOTAL_STEPS=$((TOTAL_STEPS + 1)); fi
if [[ "$RUN_META_AUTOMODEL_RUN" == "true" ]]; then TOTAL_STEPS=$((TOTAL_STEPS + 1)); fi

STEP_NO=1
run_step "$STEP_NO" "$TOTAL_STEPS" "meta-automodel-create"
echo "target filter: loto=${TARGET_LOTO}, unique_id=${TARGET_UNIQUE_ID}, ts_type=${TARGET_TS_TYPE}"
echo "meta unified_filter_json: ${UNIFIED_FILTER_JSON}"
echo "meta run mode: console_log_mode=${CONSOLE_LOG_MODE}, one_model_only=${ONE_MODEL_ONLY}, recursive_depth=${RECURSIVE_DEPTH}, max_tasks=${MAX_TASKS}, run_explain=${RUN_EXPLAIN}"

META_CMD=(
  python -m loto_forecast.cli meta-automodel-create
  --config-name "$CONFIG_NAME"
  --base-schema "$BASE_SCHEMA" --base-table "$BASE_TABLE"
  --hist-schema "$HIST_SCHEMA" --hist-table "$HIST_TABLE"
  --exog-schema "$EXOG_SCHEMA"
  --output-schema "$OUTPUT_SCHEMA" --output-table "$OUTPUT_TABLE"
  --unified-filter-json "$UNIFIED_FILTER_JSON"
  --model-name "$MODEL_NAME" --h "$HORIZON"
  --unified-group-cols-json "$UNIFIED_GROUP_COLS_JSON"
  --auto-cls-model "$AUTO_CLS_MODEL"
  --auto-h "$AUTO_H"
  --auto-loss "$AUTO_LOSS"
  --auto-valid-loss "$AUTO_VALID_LOSS"
  --auto-config-json "$AUTO_CONFIG_JSON"
  --auto-search-alg "$AUTO_SEARCH_ALG"
  --auto-num-samples "$AUTO_NUM_SAMPLES"
  --auto-backend "$AUTO_BACKEND"
  --auto-callbacks-json "$AUTO_CALLBACKS_JSON"
  --model-params-json "$MODEL_PARAMS_JSON"
  --param-space-json "$PARAM_SPACE_JSON"
  --recursive-depth "$RECURSIVE_DEPTH"
  --save-path "$SAVE_PATH"
)

if [[ "$AUTO_CPUS" != "0" ]]; then META_CMD+=(--auto-cpus "$AUTO_CPUS"); fi
if [[ "$AUTO_GPUS" != "0" ]]; then META_CMD+=(--auto-gpus "$AUTO_GPUS"); fi
if [[ "$AUTO_VERBOSE" == "true" ]]; then META_CMD+=(--auto-verbose); else META_CMD+=(--no-auto-verbose); fi
if [[ "$MAX_TASKS" != "0" ]]; then META_CMD+=(--max-tasks "$MAX_TASKS"); fi
if [[ "$UNIFIED_GROUP_VALIDATE_STRICT" == "true" ]]; then
  META_CMD+=(--unified-group-validate-strict)
else
  META_CMD+=(--no-unified-group-validate-strict)
fi
if [[ "$RUN_PREDICT" == "true" ]]; then META_CMD+=(--run-predict); else META_CMD+=(--no-run-predict); fi
if [[ "$RUN_EVALUATE" == "true" ]]; then META_CMD+=(--run-evaluate); else META_CMD+=(--no-run-evaluate); fi
if [[ "$RUN_EXPLAIN" == "true" ]]; then META_CMD+=(--run-explain); else META_CMD+=(--no-run-explain); fi
if [[ "$RUN_SAVE" == "true" ]]; then META_CMD+=(--run-save); else META_CMD+=(--no-run-save); fi
if [[ "$RUN_LOAD" == "true" ]]; then META_CMD+=(--run-load); else META_CMD+=(--no-run-load); fi
if [[ "$RUN_ANALYZE" == "true" ]]; then META_CMD+=(--run-analyze); else META_CMD+=(--no-run-analyze); fi
if [[ "$SAVE_DATASET" == "true" ]]; then META_CMD+=(--save-dataset); else META_CMD+=(--no-save-dataset); fi
if [[ "$SAVE_OVERWRITE" == "true" ]]; then META_CMD+=(--save-overwrite); else META_CMD+=(--no-save-overwrite); fi
if [[ "$LOAD_CHECK_PREDICT" == "true" ]]; then META_CMD+=(--load-check-predict); else META_CMD+=(--no-load-check-predict); fi

run_cmd "$STEP_NO" "$TOTAL_STEPS" "meta-automodel-create" "${META_CMD[@]}"
STEP_NO=$((STEP_NO + 1))

run_step "$STEP_NO" "$TOTAL_STEPS" "run-table-pyspark"
echo "source sql(pushdown): ${SOURCE_SQL}"
if [[ -n "$TRANSFORM_SQL" ]]; then echo "transform sql: ${TRANSFORM_SQL}"; fi
echo "execution backend: ${EXECUTION_BACKEND} (prefer_pandas=${PREFER_PANDAS}, dask_npartitions=${DASK_NPARTITIONS})"
SPARK_CMD=(
  python -m loto_forecast.cli run-table-pyspark
  --host "$DB_HOST" --port "$DB_PORT" --user "$DB_USER" --database "$DB_NAME"
  --source-schema "$SOURCE_SCHEMA" --source-table "$SOURCE_TABLE"
  --source-sql "$SOURCE_SQL"
  --target-schema "$TARGET_SCHEMA" --target-table "$TARGET_TABLE"
  --output-if-exists "$OUTPUT_IF_EXISTS"
  --output-parquet-path "$OUTPUT_PARQUET_PATH"
  --execution-backend "$EXECUTION_BACKEND"
)
if [[ "$DASK_NPARTITIONS" != "0" ]]; then SPARK_CMD+=(--dask-npartitions "$DASK_NPARTITIONS"); fi
if [[ -n "$TRANSFORM_SQL" ]]; then SPARK_CMD+=(--transform-sql "$TRANSFORM_SQL"); fi
if [[ "$REPARTITION" != "0" ]]; then SPARK_CMD+=(--repartition "$REPARTITION"); fi
if [[ -n "$SPARK_MASTER" ]]; then SPARK_CMD+=(--spark-master "$SPARK_MASTER"); fi
if [[ "$PREFER_PANDAS" == "true" ]]; then SPARK_CMD+=(--prefer-pandas); else SPARK_CMD+=(--no-prefer-pandas); fi
if [[ "$SKIP_ROW_COUNT" == "true" ]]; then SPARK_CMD+=(--skip-row-count); else SPARK_CMD+=(--no-skip-row-count); fi
if [[ "$SPARK_UI_ENABLED" == "true" ]]; then SPARK_CMD+=(--spark-ui-enabled); else SPARK_CMD+=(--no-spark-ui-enabled); fi
if [[ "${SPARK_SHUFFLE_PARTITIONS}" != "" ]]; then SPARK_CMD+=(--spark-shuffle-partitions "$SPARK_SHUFFLE_PARTITIONS"); fi
if [[ "${SPARK_READER_FETCHSIZE}" != "" ]]; then SPARK_CMD+=(--spark-reader-fetchsize "$SPARK_READER_FETCHSIZE"); fi
SPARK_CMD+=(--postgres-write-mode "$POSTGRES_WRITE_MODE")
if [[ "${POSTGRES_COPY_CHUNK_ROWS}" != "" ]]; then SPARK_CMD+=(--postgres-copy-chunk-rows "$POSTGRES_COPY_CHUNK_ROWS"); fi
if [[ "${POSTGRES_LOCK_TIMEOUT_MS}" != "" ]]; then SPARK_CMD+=(--postgres-lock-timeout-ms "$POSTGRES_LOCK_TIMEOUT_MS"); fi
run_cmd "$STEP_NO" "$TOTAL_STEPS" "run-table-pyspark" "${SPARK_CMD[@]}"
STEP_NO=$((STEP_NO + 1))

if [[ "$RUN_CHECK_GROUPING" == "true" ]]; then
  run_step "$STEP_NO" "$TOTAL_STEPS" "check-unified-grouping"
  run_cmd "$STEP_NO" "$TOTAL_STEPS" "check-unified-grouping" python -m loto_forecast.cli check-unified-grouping \
    --host "$DB_HOST" --port "$DB_PORT" --user "$DB_USER" --database "$DB_NAME" \
    --schema "$CHECK_SCHEMA" --table "$CHECK_TABLE" \
    --group-cols "$CHECK_GROUP_COLS" --time-col "$CHECK_TIME_COL"
  STEP_NO=$((STEP_NO + 1))
fi

if [[ "$RUN_META_AUTOMODEL_RUN" == "true" ]]; then
  run_step "$STEP_NO" "$TOTAL_STEPS" "meta-automodel-run"
  RESOLVED_CONFIG_ID="$META_RUN_CONFIG_ID"
  if [[ -z "$RESOLVED_CONFIG_ID" ]]; then
    if RESOLVED_CONFIG_ID="$(resolve_config_id_by_name "$CONFIG_NAME")"; then
      echo "meta run target config_id=${RESOLVED_CONFIG_ID} (resolved by config_name=${CONFIG_NAME})"
    else
      echo "meta run target config resolution failed for config_name=${CONFIG_NAME}; fallback to --limit ${META_LIMIT}"
      RESOLVED_CONFIG_ID=""
    fi
  else
    echo "meta run target config_id=${RESOLVED_CONFIG_ID} (from META_RUN_CONFIG_ID)"
  fi

  META_RUN_CMD=(python -m loto_forecast.cli meta-automodel-run)
  if [[ -n "$RESOLVED_CONFIG_ID" ]]; then
    META_RUN_CMD+=(--config-id "$RESOLVED_CONFIG_ID")
  else
    META_RUN_CMD+=(--limit "$META_LIMIT")
  fi
  if [[ "$META_STOP_ON_ERROR" == "true" ]]; then META_RUN_CMD+=(--stop-on-error); fi

  run_cmd "$STEP_NO" "$TOTAL_STEPS" "meta-automodel-run" "${META_RUN_CMD[@]}"
fi

echo "[done] $(date '+%Y-%m-%d %H:%M:%S') logs=$LOG_DIR run_tag=$RUN_TAG"
