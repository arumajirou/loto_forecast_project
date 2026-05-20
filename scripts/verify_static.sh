#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# Ensure only lightweight runtime dependencies + dev tools are installed.
# Do not use --locked here; distributed locks may contain mirror-specific URLs.
uv sync --extra dev

uv run --no-sync python -m compileall -q src tests tools evals scripts
uv run --no-sync ruff check src tests --no-fix
uv run --no-sync ruff format src tests --check
uv run --no-sync mypy src/loto_forecast --ignore-missing-imports
uv run --no-sync bandit -r src/loto_forecast -c pyproject.toml
