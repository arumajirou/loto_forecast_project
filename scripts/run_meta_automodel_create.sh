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

CONFIG_NAME="${CONFIG_NAME:-local_nf_run_01}"
BASE_SCHEMA="${BASE_SCHEMA:-dataset}"
BASE_TABLE="${BASE_TABLE:-loto_y_ts_unified}"
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
RECURSIVE_DEPTH="${RECURSIVE_DEPTH:-2}"
ONE_MODEL_ONLY="${ONE_MODEL_ONLY:-false}"
AUTO_LOSS="${AUTO_LOSS:-MAE}"
AUTO_VALID_LOSS="${AUTO_VALID_LOSS:-MAE}"
AUTO_SEARCH_ALG="${AUTO_SEARCH_ALG:-BasicVariantGenerator}"
AUTO_NUM_SAMPLES="${AUTO_NUM_SAMPLES:-10}"
AUTO_CPUS="${AUTO_CPUS:-0}"
AUTO_GPUS="${AUTO_GPUS:-0}"
AUTO_VERBOSE="${AUTO_VERBOSE:-true}"
AUTO_BACKEND="${AUTO_BACKEND:-optuna}"
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
SAVE_PATH="${SAVE_PATH:-./artifacts/saved_models/{run_id}}"
LOAD_CHECK_PREDICT="${LOAD_CHECK_PREDICT:-false}"
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

if [[ "$ONE_MODEL_ONLY" == "true" ]]; then
  RECURSIVE_DEPTH=1
  MAX_TASKS=1
  if [[ -z "$RUN_EXPLAIN_IS_SET" ]]; then
    RUN_EXPLAIN=false
  fi
fi

CMD=(
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

if [[ "$AUTO_CPUS" != "0" ]]; then CMD+=(--auto-cpus "$AUTO_CPUS"); fi
if [[ "$AUTO_GPUS" != "0" ]]; then CMD+=(--auto-gpus "$AUTO_GPUS"); fi
if [[ "$AUTO_VERBOSE" == "true" ]]; then CMD+=(--auto-verbose); else CMD+=(--no-auto-verbose); fi
if [[ "$MAX_TASKS" != "0" ]]; then CMD+=(--max-tasks "$MAX_TASKS"); fi
if [[ "$UNIFIED_GROUP_VALIDATE_STRICT" == "true" ]]; then
  CMD+=(--unified-group-validate-strict)
else
  CMD+=(--no-unified-group-validate-strict)
fi
if [[ "$RUN_SAVE" == "true" ]]; then CMD+=(--run-save); else CMD+=(--no-run-save); fi
if [[ "$RUN_LOAD" == "true" ]]; then CMD+=(--run-load); else CMD+=(--no-run-load); fi
if [[ "$RUN_ANALYZE" == "true" ]]; then CMD+=(--run-analyze); else CMD+=(--no-run-analyze); fi
if [[ "$SAVE_DATASET" == "true" ]]; then CMD+=(--save-dataset); else CMD+=(--no-save-dataset); fi
if [[ "$SAVE_OVERWRITE" == "true" ]]; then CMD+=(--save-overwrite); else CMD+=(--no-save-overwrite); fi
if [[ "$LOAD_CHECK_PREDICT" == "true" ]]; then CMD+=(--load-check-predict); else CMD+=(--no-load-check-predict); fi
if [[ "$RUN_PREDICT" == "true" ]]; then CMD+=(--run-predict); else CMD+=(--no-run-predict); fi
if [[ "$RUN_EVALUATE" == "true" ]]; then CMD+=(--run-evaluate); else CMD+=(--no-run-evaluate); fi
if [[ "$RUN_EXPLAIN" == "true" ]]; then CMD+=(--run-explain); else CMD+=(--no-run-explain); fi

"${CMD[@]}"
