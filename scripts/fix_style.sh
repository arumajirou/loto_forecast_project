#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

uv sync --extra dev
uv run --no-sync ruff check src tests --fix
uv run --no-sync ruff format src tests
