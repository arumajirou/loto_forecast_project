# v12 Setup and Screenshot Capture Commands

## Apply v12
```bash
cd /mnt/e/env/fc

mkdir -p old
TS="$(date +%Y%m%d_%H%M%S)"

rsync -a --info=progress2 \
  --exclude '.venv/' \
  --exclude 'node_modules/' \
  --exclude 'artifacts/' \
  --exclude 'logs/' \
  --exclude 'outputs/' \
  --exclude 'reports/' \
  --exclude 'traces/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  loto_forecast_project/ \
  "old/loto_forecast_project_before_v12_repair_${TS}/"

echo "${TS}" | tee old/latest_v12_backup_ts.txt

rm -rf loto_forecast_project.v12_repaired
mkdir -p loto_forecast_project.v12_repaired
unzip -q zips/loto_forecast_project_uv_repaired_v12.zip -d loto_forecast_project.v12_repaired

rsync -a --delete --info=progress2 --human-readable \
  --exclude '.env' \
  --exclude '.venv/' \
  loto_forecast_project.v12_repaired/loto_forecast_project/ \
  loto_forecast_project/
```

## Clean stale collectors
```bash
cd /mnt/e/env/fc/loto_forecast_project

pkill -f "scripts/collect_browser_observability.py" 2>/dev/null || true
pkill -f "playwright/driver" 2>/dev/null || true
pkill -f "chrome-headless-shell" 2>/dev/null || true
```

## Run with progress output
```bash
cd /mnt/e/env/fc/loto_forecast_project

export UV_LINK_MODE=copy
export LOTO_UV_ENV_MODE=browser
export LOTO_UV_CLEAR_VENV=0
export LOTO_PLAYWRIGHT_INSTALL=0

./scripts/start_dashboard_and_capture_screenshots.sh --max-clicks 80 --max-depth 3
```

## Watch progress files from another terminal
```bash
cd /mnt/e/env/fc/loto_forecast_project
LATEST="$(find artifacts/observability/browser_runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tail -f "${LATEST}/progress.jsonl"
```
