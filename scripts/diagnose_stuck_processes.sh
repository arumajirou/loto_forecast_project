#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "== cwd =="
pwd

echo
echo "== uv/python =="
command -v uv || true
uv --version || true
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python --version || true
fi

echo
echo "== process: streamlit/playwright/chromium/uv =="
ps -eo pid,ppid,stat,etimes,cmd | grep -E 'streamlit|operations_dashboard|playwright|chromium|chrome|uv run' | grep -v grep || true

echo
echo "== listening ports =="
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) | grep -E '8505|8506|streamlit|python' || true

echo
echo "== artifacts logs =="
find artifacts -maxdepth 4 -type f \( -name '*.log' -o -name '*.pid' -o -name '*.json' -o -name '*.jsonl' \) 2>/dev/null | sort | tail -80 || true

echo
echo "== recent dashboard logs =="
for f in \
  artifacts/automation/logs/dashboard_autostart.log \
  artifacts/observability/launcher/streamlit_8505.log
do
  if [[ -f "$f" ]]; then
    echo "----- $f -----"
    tail -120 "$f" || true
  fi
done


echo
echo "Hint: run bash ./scripts/kill_loto_observability_processes.sh to clean stopped Playwright/dashboard processes."
