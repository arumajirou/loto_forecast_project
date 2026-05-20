# V6 Patch Notes

## Conclusion

V6 fixes the remaining runtime setup issues found after V5 static verification succeeded.

## Fixes

- Added `psycopg[binary]>=3.1.18` to base dependencies because dashboard imports `src/resources/db/postgres_copy.py`, which imports `psycopg`.
- Made `scripts/setup_uv.sh` non-interactive by clearing `.venv` by default before `uv venv`.
- Added `LOTO_UV_CLEAR_VENV=0` escape hatch for users who intentionally want to reuse `.venv`.
- Kept static mode lightweight. Static mode still avoids `neuralforecast`, `torch`, `nvidia-*`, and `cuda-*` dependencies.
- Kept `uv.lock` out of the distributed ZIP.

## Recommended commands

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
./scripts/setup_uv.sh
./scripts/repair_static.sh

LOTO_UV_ENV_MODE=dashboard ./scripts/setup_uv.sh
PYTHONPATH=src uv run --no-sync streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py
```

## Safety

No DB write, `db-init` real apply, training, E2E, or browser automation is performed by these setup scripts.
