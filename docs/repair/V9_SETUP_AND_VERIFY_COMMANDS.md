# V9 適用・UI/UX確認コマンド

## 1. ZIP配置
```bash
/mnt/e/env/fc/zips/loto_forecast_project_uv_repaired_v9.zip
```

## 2. 反映
```bash
cd /mnt/e/env/fc

mkdir -p old
TS="$(date +%Y%m%d_%H%M%S)"

if [ -d loto_forecast_project ]; then
  rsync -a --info=progress2     --exclude '.venv/'     --exclude 'node_modules/'     --exclude 'artifacts/'     --exclude 'logs/'     --exclude 'outputs/'     --exclude 'reports/'     --exclude 'traces/'     --exclude '.mypy_cache/'     --exclude '.pytest_cache/'     --exclude '__pycache__/'     --exclude '*.pyc'     loto_forecast_project/     "old/loto_forecast_project_before_v9_repair_${TS}/"
  echo "${TS}" | tee old/latest_v9_backup_ts.txt
fi

rm -rf loto_forecast_project.v9_repaired
mkdir -p loto_forecast_project.v9_repaired
unzip -q zips/loto_forecast_project_uv_repaired_v9.zip -d loto_forecast_project.v9_repaired

rsync -a --delete --info=progress2 --human-readable   --exclude '.env'   --exclude '.venv/'   loto_forecast_project.v9_repaired/loto_forecast_project/   loto_forecast_project/
```

## 3. 静的検査
```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src
export UV_LINK_MODE=copy

./scripts/setup_uv.sh
./scripts/repair_static.sh
```

## 4. DB_PASSWORD設定
```bash
cd /mnt/e/env/fc/loto_forecast_project
cp -n .env.example .env
chmod 600 .env
nano .env
```

`.env` に次を設定してください。

```text
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=loto
DB_NAME=loto
DB_PASSWORD=<実パスワード>
```

## 5. dashboard起動
```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
LOTO_UV_ENV_MODE=dashboard LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh

./run_operations_dashboard.sh
```

ブラウザで開くURL:

```text
http://localhost:8505
```

## 6. 確認ポイント
- サイドバーの `表示パネル(高速モード)` の先頭が `NeuralForecast Cockpit`
- DB未接続でも `はじめる / DB接続 / 実行計画 / 観測・診断 / 重複/棚卸 / 詳細ラボ` が表示される
- DB接続失敗詳細が折りたたみ表示になる
- `DB_PASSWORD未設定` が早期検知される
- 既存詳細UIは `詳細ラボ` タブに残る

## 7. 観測収集
```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_LINK_MODE=copy
LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh
uv run --no-sync playwright install chromium

LOTO_UV_CLEAR_VENV=0 ./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2
```

## 8. ロールバック
```bash
cd /mnt/e/env/fc
TS="$(cat old/latest_v9_backup_ts.txt)"
rm -rf loto_forecast_project
cp -a "old/loto_forecast_project_before_v9_repair_${TS}" loto_forecast_project
```
