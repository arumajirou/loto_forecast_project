#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export LOTO_DB_PATH="${LOTO_DB_PATH:-$ROOT_DIR/data/registry.sqlite}"
HOST="${ASYNC_API_HOST:-0.0.0.0}"
PORT="${ASYNC_API_PORT:-8000}"

echo "[async-api] LOTO_DB_PATH=$LOTO_DB_PATH"
echo "[async-api] start: http://$HOST:$PORT"
python -m uvicorn loto_forecast.api.server:app --host "$HOST" --port "$PORT"
