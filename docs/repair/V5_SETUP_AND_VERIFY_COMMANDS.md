# v5 適用・検証コマンド

## 1. ZIP配置

```bash
/mnt/e/env/fc/zips/loto_forecast_project_uv_repaired_v5.zip
```

## 2. 適用

```bash
cd /mnt/e/env/fc

mkdir -p old
TS="$(date +%Y%m%d_%H%M%S)"

# 既存ディレクトリがあれば、重い生成物と .venv を除外してバックアップ
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
    "old/loto_forecast_project_before_v5_repair_${TS}/"
  echo "${TS}" | tee old/latest_v5_backup_ts.txt
fi

rm -rf loto_forecast_project.v5_repaired
mkdir -p loto_forecast_project.v5_repaired
unzip -q zips/loto_forecast_project_uv_repaired_v5.zip -d loto_forecast_project.v5_repaired

rsync -a --delete --info=progress2 --human-readable \
  --exclude '.env' \
  --exclude '.venv/' \
  loto_forecast_project.v5_repaired/loto_forecast_project/ \
  loto_forecast_project/
```

## 3. v4由来の古いlockと環境を削除

```bash
cd /mnt/e/env/fc/loto_forecast_project
rm -f uv.lock
rm -rf .venv
```

## 4. 静的検査用の軽量uv環境

```bash
cd /mnt/e/env/fc/loto_forecast_project

export UV_LINK_MODE=copy
./scripts/setup_uv.sh
```

このコマンドは既定で `LOTO_UV_ENV_MODE=static` を使い、`neuralforecast` / `torch` / `nvidia-*` / `cuda-*` をインストールしません。

## 5. style自動修正 + 静的検査

```bash
cd /mnt/e/env/fc/loto_forecast_project

export PYTHONPATH=src
export UV_LINK_MODE=copy

./scripts/repair_static.sh
```

個別に実行する場合:

```bash
./scripts/fix_style.sh
./scripts/verify_static.sh
```

## 6. dashboardだけ使う場合

```bash
cd /mnt/e/env/fc/loto_forecast_project

rm -f uv.lock
export UV_LINK_MODE=copy
LOTO_UV_ENV_MODE=dashboard ./scripts/setup_uv.sh
PYTHONPATH=src uv run streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py
```

## 7. 学習まで含むフル環境

時間・容量・ネットワーク負荷が大きいので、静的検査が通ってから実行する。

```bash
cd /mnt/e/env/fc/loto_forecast_project

rm -f uv.lock
export UV_LINK_MODE=copy
LOTO_UV_ENV_MODE=full ./scripts/setup_uv.sh
```

## 8. ロールバック

```bash
cd /mnt/e/env/fc
TS="$(cat old/latest_v5_backup_ts.txt)"
rm -rf loto_forecast_project
cp -a "old/loto_forecast_project_before_v5_repair_${TS}" loto_forecast_project
```
