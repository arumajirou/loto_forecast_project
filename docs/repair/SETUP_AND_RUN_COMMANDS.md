# SETUP_AND_RUN_COMMANDS

以下は、修復済みZIPを `zips` ディレクトリに置いた後、`z@az:/mnt/e/env/fc` で展開・セットアップ・確認するためのコマンドです。

## 0. 前提

- ZIPファイル名: `loto_forecast_project_repaired.zip`
- 配置先: `/mnt/e/env/fc/zips/loto_forecast_project_repaired.zip`
- 現在の作業ルート: `/mnt/e/env/fc`
- 既存プロジェクト: `/mnt/e/env/fc/loto_forecast_project`

## 1. 配置確認

```bash
cd /mnt/e/env/fc
ls -lh zips/loto_forecast_project_repaired.zip
```

## 2. 既存プロジェクトのバックアップ

```bash
cd /mnt/e/env/fc
mkdir -p old
TS="$(date +%Y%m%d_%H%M%S)"
tar -czf "old/loto_forecast_project_before_repair_${TS}.tar.gz" loto_forecast_project
ls -lh "old/loto_forecast_project_before_repair_${TS}.tar.gz"
```

## 3. ZIPを一時展開

```bash
cd /mnt/e/env/fc
rm -rf loto_forecast_project.repaired
mkdir -p loto_forecast_project.repaired
unzip -q zips/loto_forecast_project_repaired.zip -d loto_forecast_project.repaired
find loto_forecast_project.repaired -maxdepth 2 -type d | sort
```

## 4. 修復版を既存ディレクトリへ反映

`.env` と `.venv` はローカル環境依存のため保護します。

```bash
cd /mnt/e/env/fc
rsync -a --delete \
  --exclude '.env' \
  --exclude '.venv/' \
  loto_forecast_project.repaired/loto_forecast_project/ \
  loto_forecast_project/
```

## 5. 不要物が残っていないか確認

```bash
cd /mnt/e/env/fc/loto_forecast_project

find . -name '*:Zone.Identifier' -o -name '=70.0.0'
test ! -d node_modules && echo "OK: node_modules removed"
test ! -d .mypy_cache && echo "OK: .mypy_cache removed"
test ! -d .pytest_cache && echo "OK: .pytest_cache removed"
test ! -d artifacts && echo "OK: artifacts removed"
test ! -d logs && echo "OK: logs removed"
```

## 6. uv 仮想環境セットアップ

```bash
cd /mnt/e/env/fc/loto_forecast_project

uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"
./scripts/setup_uv.sh
```

手動で行う場合:

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"
uv venv --python 3.11
uv sync --extra dev --locked
```

`uv sync --locked` が lockfile 不整合で失敗する場合は、依存変更の有無を確認した上で `uv lock` を実行し、再度 `uv sync --extra dev --locked` を実行してください。

## 7. `.env` 作成

```bash
cd /mnt/e/env/fc/loto_forecast_project
cp -n .env.example .env
chmod 600 .env
```

`.env` を開いて、少なくとも `DB_PASSWORD=` をローカル値に設定してください。パスワードはコマンドライン引数に書かないでください。

```bash
nano .env
```

## 8. 静的検査

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src

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

## 10. DB初期化の確認だけ行う

これは dry-run なのでSQLを実行しません。

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src

uv run python -m loto_forecast.cli db-init --dry-run
```

## 11. DB初期化を実適用する場合

バックアップ確認後にだけ実行してください。

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src

export LOTO_ALLOW_DB_INIT=1
uv run python -m loto_forecast.cli db-init --yes-i-understand-db-init-may-write
unset LOTO_ALLOW_DB_INIT
```

## 12. dashboard 起動確認

```bash
cd /mnt/e/env/fc/loto_forecast_project
./run_operations_dashboard.sh
```

または:

```bash
cd /mnt/e/env/fc/loto_forecast_project
PYTHONPATH=src uv run streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py
```

## 13. grid-run を実行する場合

長時間・DB書き込みの可能性があるため、上限・ログ・対象gridを確認してから実行してください。

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src

export LOTO_ALLOW_GRID_RUN=1
uv run python -m loto_forecast.cli grid-run --grid-id nf_grid_001
unset LOTO_ALLOW_GRID_RUN
```

## 14. ロールバック

```bash
cd /mnt/e/env/fc
rm -rf loto_forecast_project
tar -xzf old/loto_forecast_project_before_repair_<TS>.tar.gz
```

`<TS>` はバックアップ作成時のタイムスタンプに置き換えてください。


## v6 note: non-interactive uv environment rebuild

`./scripts/setup_uv.sh` now clears `.venv` by default to avoid the interactive uv prompt when switching from `static` to `dashboard` or `full` mode. Set `LOTO_UV_CLEAR_VENV=0` only when you explicitly want to reuse the existing environment.

Dashboard mode now includes `psycopg[binary]` through base dependencies because `src/resources/db/postgres_copy.py` imports the `psycopg` package at runtime.
