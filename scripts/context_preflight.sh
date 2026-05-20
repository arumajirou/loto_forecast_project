#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT}

echo "== AGENTS =="
test -f AGENTS.md && echo "AGENTS: OK" || echo "AGENTS: MISSING"

echo
echo "== Context docs =="
for f in \
  docs/context/00_context_index.md \
  docs/context/01_execution_contract.md \
  docs/context/02_decision_policy.md \
  docs/context/03_context_packet.md \
  docs/context/04_tooling_scope.md
do
  test -f "$f" && echo "OK: $f" || echo "MISSING: $f"
done

echo
echo "== MCP =="
codex mcp list || true

echo
echo "== Project-local Skills =="
find .agents/skills -maxdepth 2 -name SKILL.md | sort || true

echo
echo "== Global Skills =="
find "$HOME/.agents/skills" -maxdepth 3 -name SKILL.md 2>/dev/null | sort || true

echo
echo "== Harness =="
test -f tests/e2e/operations_dashboard_ui_check.mjs && echo "E2E: OK" || echo "E2E: MISSING"
test -f tests/streamlit/test_operations_dashboard_apptest.py && echo "AppTest: OK" || echo "AppTest: MISSING"
test -f tests/unit/test_operations_dashboard_ui_helpers.py && echo "Unit: OK" || echo "Unit: MISSING"

echo
echo "== Trace files =="
for f in \
  artifacts/logs/coverage_matrix.md \
  artifacts/logs/action_inventory.json \
  artifacts/logs/dynamic_trace.jsonl \
  artifacts/logs/dynamic_trace.csv \
  artifacts/logs/db_observation.json \
  artifacts/logs/file_observation.json
do
  test -f "$f" && echo "OK: $f" || echo "MISSING: $f"
done
