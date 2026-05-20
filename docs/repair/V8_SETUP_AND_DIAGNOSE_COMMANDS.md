# v8 適用・dashboard起動診断コマンド

## 1. v8 ZIP配置

```bash
/mnt/e/env/fc/zips/loto_forecast_project_uv_repaired_v8.zip
```

## 2. 反映

```bash
cd /mnt/e/env/fc

mkdir -p old
TS="$(date +%Y%m%d_%H%M%S)"

if [ -d loto_forecast_project ]; then
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
    "old/loto_forecast_project_before_v8_repair_${TS}/"
  echo "${TS}" | tee old/latest_v8_backup_ts.txt
fi

rm -rf loto_forecast_project.v8_repaired
mkdir -p loto_forecast_project.v8_repaired
unzip -q zips/loto_forecast_project_uv_repaired_v8.zip -d loto_forecast_project.v8_repaired

rsync -a --delete --info=progress2 --human-readable \
  --exclude '.env' \
  --exclude '.venv/' \
  loto_forecast_project.v8_repaired/loto_forecast_project/ \
  loto_forecast_project/
```

## 3. 既存の停止中/起動中ジョブを止める

```bash
cd /mnt/e/env/fc/loto_forecast_project
jobs || true
kill %1 2>/dev/null || true
pkill -f "streamlit run .*operations_dashboard.py" 2>/dev/null || true
```

## 4. dashboard環境を作る

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy

LOTO_UV_ENV_MODE=dashboard LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh
PYTHONPATH=src uv run --no-sync python scripts/check_dashboard_import.py
```

## 5. アプリを単独起動

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
./run_operations_dashboard.sh
```

ブラウザでは次を開きます。

```text
http://localhost:8505
```

Windows側ブラウザで開けない場合は、WSL側に表示された Network URL も試してください。

## 6. 表示されない場合の診断

別ターミナルで:

```bash
cd /mnt/e/env/fc/loto_forecast_project
./scripts/diagnose_dashboard_startup.sh
```

## 7. 観測収集つき起動

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
export LOTO_DASHBOARD_READY_TIMEOUT=240

LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh
uv run --no-sync playwright install chromium

LOTO_UV_CLEAR_VENV=0 ./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2
```
