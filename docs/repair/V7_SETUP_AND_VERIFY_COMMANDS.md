# v7 適用・観測・検証コマンド

## 1. v7 ZIPを配置

```bash
/mnt/e/env/fc/zips/loto_forecast_project_uv_repaired_v7.zip
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
    "old/loto_forecast_project_before_v7_repair_${TS}/"
  echo "${TS}" | tee old/latest_v7_backup_ts.txt
fi

rm -rf loto_forecast_project.v7_repaired
mkdir -p loto_forecast_project.v7_repaired
unzip -q zips/loto_forecast_project_uv_repaired_v7.zip -d loto_forecast_project.v7_repaired

rsync -a --delete --info=progress2 --human-readable \
  --exclude '.env' \
  --exclude '.venv/' \
  loto_forecast_project.v7_repaired/loto_forecast_project/ \
  loto_forecast_project/
```

## 3. 静的検査

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src
export UV_LINK_MODE=copy

./scripts/setup_uv.sh
./scripts/repair_static.sh
```

## 4. dashboard + ブラウザ観測環境

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
LOTO_UV_ENV_MODE=browser ./scripts/setup_uv.sh
uv run --no-sync playwright install chromium
```

## 5. dashboard起動・スクリーンショット/ログ/trace収集

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2
```

## 6. 起動済みdashboardへ収集だけ実行

```bash
cd /mnt/e/env/fc/loto_forecast_project
PYTHONPATH=src uv run --no-sync python scripts/collect_browser_observability.py \
  --url http://localhost:8505 \
  --max-clicks 40 \
  --max-depth 2
```

## 7. アプリ内確認

dashboardを開き、サイドバーの `表示パネル(高速モード)` から `観測・診断` を選択する。

## 8. 保存先

```bash
find artifacts/observability -maxdepth 4 -type f | sort
```

## 9. ロールバック

```bash
cd /mnt/e/env/fc
TS="$(cat old/latest_v7_backup_ts.txt)"
rm -rf loto_forecast_project
cp -a "old/loto_forecast_project_before_v7_repair_${TS}" loto_forecast_project
```
