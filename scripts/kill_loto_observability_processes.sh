#!/usr/bin/env bash
set -euo pipefail

PATTERN='scripts/collect_browser_observability.py|playwright/driver|chrome-headless-shell|streamlit run .*operations_dashboard.py'

echo "== matching loto observability/dashboard processes before cleanup =="
ps -eo pid,ppid,stat,etimes,cmd | grep -E "${PATTERN}" | grep -v grep || true

PIDS="$(ps -eo pid,cmd | grep -E "${PATTERN}" | grep -v grep | awk '{print $1}' || true)"
if [[ -z "${PIDS}" ]]; then
  echo "OK: no loto observability/dashboard processes found"
  exit 0
fi

echo "== graceful terminate running processes =="
# Do not CONT stopped Playwright/node processes first; that can emit noisy EPIPE traces.
for pid in ${PIDS}; do
  stat="$(ps -o stat= -p "${pid}" 2>/dev/null | awk '{print $1}' || true)"
  if [[ "${stat}" == T* ]]; then
    continue
  fi
  kill -TERM "${pid}" 2>/dev/null || true
done

sleep 2

echo "== force kill remaining/stopped processes =="
PIDS2="$(ps -eo pid,cmd | grep -E "${PATTERN}" | grep -v grep | awk '{print $1}' || true)"
for pid in ${PIDS2}; do
  kill -KILL "${pid}" 2>/dev/null || true
done

sleep 1

echo "== matching loto observability/dashboard processes after cleanup =="
ps -eo pid,ppid,stat,etimes,cmd | grep -E "${PATTERN}" | grep -v grep || echo "OK: cleaned"
