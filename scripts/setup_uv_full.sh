#!/usr/bin/env bash
set -euo pipefail

# Full runtime setup for dashboard / model training. This can download PyTorch
# and ML dependencies. Static checks should use scripts/setup_uv.sh instead.
export LOTO_UV_ENV_MODE="${LOTO_UV_ENV_MODE:-full}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_uv.sh"
