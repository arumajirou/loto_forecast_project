# uv セットアップ手順

## 結論

このプロジェクトの Python 仮想環境は `uv` で作成・同期します。`.venv` を直接 activate せず、原則として `uv run ...` で実行してください。

## 前提

- 作業ディレクトリ: `/mnt/e/env/fc/loto_forecast_project`
- Python: `.python-version` により 3.11 を既定にします
- 仮想環境: `.venv`
- 依存関係: `pyproject.toml` と `uv.lock`

## uv インストール

```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
```

## セットアップ

```bash
cd /mnt/e/env/fc/loto_forecast_project
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cpu}"
./scripts/setup_uv.sh
```

GPU/CUDA版の torch を明示的に使う場合は、環境に合わせて `UV_TORCH_BACKEND` を変更してから実行してください。

```bash
export UV_TORCH_BACKEND=auto
./scripts/setup_uv.sh
```

## 実行例

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src

uv run python -m compileall -q src tests tools evals scripts
uv run pytest tests/unit -v --tb=short --no-cov
uv run ruff check src tests --no-fix
uv run mypy src/loto_forecast --ignore-missing-imports
```

## DB 初期化 dry-run

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli db-init --dry-run
```

## DB 初期化の実適用

バックアップ確認後だけ実行してください。

```bash
export LOTO_ALLOW_DB_INIT=1
PYTHONPATH=src uv run python -m loto_forecast.cli db-init --yes-i-understand-db-init-may-write
unset LOTO_ALLOW_DB_INIT
```

## dashboard

```bash
./run_operations_dashboard.sh
```

または:

```bash
PYTHONPATH=src uv run streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py
```

## ロールバック

`uv` 化で問題が出た場合も、`.venv` と lockfile は削除して再生成できます。

```bash
rm -rf .venv
uv venv --python 3.11
uv sync --extra dev --locked
```

`uv.lock` 自体を更新する必要がある場合は、依存変更の理由を記録してから実行してください。

```bash
uv lock
uv sync --extra dev --locked
```
