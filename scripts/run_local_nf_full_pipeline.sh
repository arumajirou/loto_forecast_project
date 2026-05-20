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

RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-7200}"
PROGRESS_HEARTBEAT_SECONDS="${PROGRESS_HEARTBEAT_SECONDS:-10}"
CONSOLE_LOG_MODE="${CONSOLE_LOG_MODE:-all}" # all | progress
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/pipeline_runs}"
mkdir -p "$LOG_DIR"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
STALE_PROCESS_POLICY="${STALE_PROCESS_POLICY:-kill}"  # kill | warn | ignore
ONE_MODEL_ONLY="${ONE_MODEL_ONLY:-false}"

RUN_LOCAL_SCRIPT="${RUN_LOCAL_SCRIPT:-$ROOT_DIR/scripts/run_local_nf_meta_create_and_pyspark.sh}"
RUN_MODEL_OPS_AFTER="${RUN_MODEL_OPS_AFTER:-true}"

# Performance-oriented defaults (overridable)
EXECUTION_BACKEND="${EXECUTION_BACKEND:-auto}"
PREFER_PANDAS="${PREFER_PANDAS:-false}"
POSTGRES_WRITE_MODE="${POSTGRES_WRITE_MODE:-copy}"
POSTGRES_COPY_CHUNK_ROWS="${POSTGRES_COPY_CHUNK_ROWS:-50000}"
POSTGRES_LOCK_TIMEOUT_MS="${POSTGRES_LOCK_TIMEOUT_MS:-5000}"
RUN_CHECK_GROUPING="${RUN_CHECK_GROUPING:-true}"
RUN_META_AUTOMODEL_RUN="${RUN_META_AUTOMODEL_RUN:-true}"
META_STOP_ON_ERROR="${META_STOP_ON_ERROR:-true}"

MODEL_OPS_RUN_ID="${MODEL_OPS_RUN_ID:-}"
MODEL_OPS_CONFIG_NAME="${MODEL_OPS_CONFIG_NAME:-${CONFIG_NAME:-local_nf_run_01}}"
MODEL_OPS_CONFIG_ID="${MODEL_OPS_CONFIG_ID:-}"
MODEL_OPS_RUN_STATUS="${MODEL_OPS_RUN_STATUS:-success}"
MODEL_OPS_SOURCE_PATH="${MODEL_OPS_SOURCE_PATH:-}"
MODEL_OPS_SAVE_PATH="${MODEL_OPS_SAVE_PATH:-$ROOT_DIR/artifacts/saved_models/{run_id}}"
MODEL_OPS_RUN_SAVE="${MODEL_OPS_RUN_SAVE:-true}"
MODEL_OPS_RUN_LOAD="${MODEL_OPS_RUN_LOAD:-true}"
MODEL_OPS_RUN_ANALYZE="${MODEL_OPS_RUN_ANALYZE:-true}"
MODEL_OPS_SAVE_DATASET="${MODEL_OPS_SAVE_DATASET:-false}"
MODEL_OPS_SAVE_OVERWRITE="${MODEL_OPS_SAVE_OVERWRITE:-true}"
MODEL_OPS_LOAD_CHECK_PREDICT="${MODEL_OPS_LOAD_CHECK_PREDICT:-false}"
MODEL_OPS_INSAMPLE_STEP_SIZE="${MODEL_OPS_INSAMPLE_STEP_SIZE:-1}"
MODEL_OPS_AUTO_RESOLVE_RUN_ID="${MODEL_OPS_AUTO_RESOLVE_RUN_ID:-true}"

if [[ "$CONSOLE_LOG_MODE" != "all" && "$CONSOLE_LOG_MODE" != "progress" ]]; then
  echo "invalid CONSOLE_LOG_MODE=${CONSOLE_LOG_MODE}. use all|progress"
  exit 1
fi

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

find_stale_stopped_pids() {
  ps -eo pid,ppid,stat,cmd \
    | awk -v self="$$" '
      $1 == self {next}
      $3 ~ /^T/ {
        cmd = $0
        if (cmd ~ /python -m loto_forecast\.cli (build-unified-dataset|run-table-pyspark|meta-automodel-run)/ ||
            cmd ~ /scripts\/run_local_nf_full_pipeline\.sh/ ||
            cmd ~ /scripts\/run_local_nf_meta_create_and_pyspark\.sh/) {
          print $1
        }
      }
    '
}

handle_stale_processes() {
  local pids
  pids="$(find_stale_stopped_pids || true)"
  if [[ -z "${pids// }" ]]; then
    return 0
  fi

  echo "[preflight] detected stale stopped processes:"
  for p in $pids; do
    ps -p "$p" -o pid,ppid,stat,etime,cmd --no-headers || true
  done

  case "$STALE_PROCESS_POLICY" in
    kill)
      echo "[preflight] policy=kill -> killing stale stopped processes"
      for p in $pids; do
        if kill -0 "$p" 2>/dev/null; then
          kill -KILL "$p" || true
        fi
      done
      ;;
    warn)
      echo "[preflight] policy=warn -> continuing. if hang occurs, re-run with STALE_PROCESS_POLICY=kill"
      ;;
    ignore)
      ;;
    *)
      echo "[preflight] unknown STALE_PROCESS_POLICY=${STALE_PROCESS_POLICY} (use kill|warn|ignore)."
      exit 1
      ;;
  esac
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

TOTAL_STEPS=1
if [[ "$RUN_MODEL_OPS_AFTER" == "true" ]]; then TOTAL_STEPS=$((TOTAL_STEPS + 1)); fi

STEP_NO=1
handle_stale_processes
echo "[$STEP_NO/$TOTAL_STEPS] $(date '+%Y-%m-%d %H:%M:%S') local-train-pipeline"
echo "pipeline target: loto=${TARGET_LOTO:-bingo5}, unique_id=${TARGET_UNIQUE_ID:-N1}, ts_type=${TARGET_TS_TYPE:-raw}"
echo "pipeline config: config_name=${CONFIG_NAME:-local_nf_run_01}, execution_backend=${EXECUTION_BACKEND}, prefer_pandas=${PREFER_PANDAS}"
echo "pipeline mode: console_log_mode=${CONSOLE_LOG_MODE}, one_model_only=${ONE_MODEL_ONLY}"
run_cmd "$STEP_NO" "$TOTAL_STEPS" "local-train-pipeline" bash "$RUN_LOCAL_SCRIPT"
STEP_NO=$((STEP_NO + 1))

if [[ "$RUN_MODEL_OPS_AFTER" == "true" ]]; then
  echo "[$STEP_NO/$TOTAL_STEPS] $(date '+%Y-%m-%d %H:%M:%S') model-save-load-analyze"
  MODEL_OPS_CMD=(
    env
    "AUTO_RESOLVE_RUN_ID=${MODEL_OPS_AUTO_RESOLVE_RUN_ID}"
    "RUN_ID=${MODEL_OPS_RUN_ID}"
    "CONFIG_NAME=${MODEL_OPS_CONFIG_NAME}"
    "CONFIG_ID=${MODEL_OPS_CONFIG_ID}"
    "RUN_STATUS=${MODEL_OPS_RUN_STATUS}"
    "SOURCE_PATH=${MODEL_OPS_SOURCE_PATH}"
    "SAVE_PATH=${MODEL_OPS_SAVE_PATH}"
    "RUN_SAVE=${MODEL_OPS_RUN_SAVE}"
    "RUN_LOAD=${MODEL_OPS_RUN_LOAD}"
    "RUN_ANALYZE=${MODEL_OPS_RUN_ANALYZE}"
    "SAVE_DATASET=${MODEL_OPS_SAVE_DATASET}"
    "SAVE_OVERWRITE=${MODEL_OPS_SAVE_OVERWRITE}"
    "LOAD_CHECK_PREDICT=${MODEL_OPS_LOAD_CHECK_PREDICT}"
    "INSAMPLE_STEP_SIZE=${MODEL_OPS_INSAMPLE_STEP_SIZE}"
    bash "$ROOT_DIR/scripts/run_model_save_load_analyze.sh"
  )
  run_cmd "$STEP_NO" "$TOTAL_STEPS" "model-save-load-analyze" "${MODEL_OPS_CMD[@]}"
fi

echo "[done] $(date '+%Y-%m-%d %H:%M:%S') logs=$LOG_DIR run_tag=$RUN_TAG"
