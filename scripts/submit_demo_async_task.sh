#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
NUM_CPUS="${NUM_CPUS:-2}"
NUM_GPUS="${NUM_GPUS:-0}"

curl -sS -X POST "$API_BASE/tasks/submit" \
  -H "Content-Type: application/json" \
  -d "{
    \"kind\":\"train\",
    \"callable\":\"loto_forecast.pipeline_hooks:demo_train_and_predict\",
    \"params\":{\"resource_interval_s\":1.0, \"dataset_id\":\"demo\", \"n\":240},
    \"num_cpus\":$NUM_CPUS,
    \"num_gpus\":$NUM_GPUS
  }"

echo
