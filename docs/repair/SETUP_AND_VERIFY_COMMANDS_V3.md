# loto_forecast_project v3 展開・セットアップ・検証コマンド

## 1. ZIP配置

```bash
cd /mnt/e/env/fc
ls -lh zips/loto_forecast_project_uv_repaired_v3.zip
```

## 2. 既存プロジェクトのバックアップ

```bash
cd /mnt/e/env/fc
mkdir -p old
TS="$(date +%Y%m%d_%H%M%S)"
tar -czf "old/loto_forecast_project_before_v3_repair_${TS}.tar.gz" loto_forecast_project
ls -lh "old/loto_forecast_project_before_v3_repair_${TS}.tar.gz"
```

## 3. 一時展開

```bash
cd /mnt/e/env/fc
rm -rf loto_forecast_project.v3_repaired
mkdir -p loto_forecast_project.v3_repaired
unzip -q zips/loto_forecast_project_uv_repaired_v3.zip -d loto_forecast_project.v3_repaired
find loto_forecast_project.v3_repaired -maxdepth 2 -type d | sort
```

## 4. 既存ディレクトリへ反映

`.env` と `.venv` は保護します。

```bash
cd /mnt/e/env/fc
rsync -a --delete   --exclude '.env'   --exclude '.venv/'   loto_forecast_project.v3_repaired/loto_forecast_project/   loto_forecast_project/
```

## 5. uv確認

```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
uv --version
```

## 6. uv同期

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"
./scripts/setup_uv.sh
```

## 7. .env作成

```bash
cd /mnt/e/env/fc/loto_forecast_project
cp -n .env.example .env
chmod 600 .env
nano .env
```

`DB_PASSWORD` は `.env` に設定してください。CLI引数にパスワードを書かないでください。

## 8. 静的検証

```bash
cd /mnt/e/env/fc/loto_forecast_project
./scripts/verify_static.sh
```

個別に実行する場合:

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"

uv run python -m compileall -q src tests tools evals scripts
uv run ruff check src tests --no-fix
uv run ruff format src tests --check
uv run mypy src/loto_forecast --ignore-missing-imports
uv run bandit -r src/loto_forecast -c pyproject.toml
```

## 9. 単体テスト

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src
uv run pytest tests/unit -v --tb=short --no-cov
```

## 10. db-init dry-run

SQLは実行されません。

```bash
cd /mnt/e/env/fc/loto_forecast_project
PYTHONPATH=src uv run python -m loto_forecast.cli db-init --dry-run
```

## 11. db-init 実適用

バックアップ確認後だけ実行してください。

```bash
cd /mnt/e/env/fc/loto_forecast_project
export LOTO_ALLOW_DB_INIT=1
PYTHONPATH=src uv run python -m loto_forecast.cli db-init --yes-i-understand-db-init-may-write
unset LOTO_ALLOW_DB_INIT
```

## 12. dashboard

```bash
cd /mnt/e/env/fc/loto_forecast_project
./run_operations_dashboard.sh
```

## 13. ロールバック

```bash
cd /mnt/e/env/fc
rm -rf loto_forecast_project
tar -xzf old/loto_forecast_project_before_v3_repair_<TS>.tar.gz
```
