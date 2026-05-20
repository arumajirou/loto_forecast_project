#!/usr/bin/env bash
set -euo pipefail

# One-command local repair for style drift, followed by static verification.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

./scripts/fix_style.sh
./scripts/verify_static.sh
