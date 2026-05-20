# v4 patch notes

## 結論

v3 適用後のユーザー実行ログで残った Ruff 整形差分と setup_uv.sh の実行権限を修正しました。

## 修正内容

- `scripts/setup_uv.sh` に実行権限を付与
- `scripts/fix_style.sh` を追加
- `src/loto_forecast/api/streamlit/ui/wizard.py` の Ruff 整形差分を修正
- `src/loto_forecast/services/task_runner.py` の import block を Ruff/isort 形式へ修正
- `src/loto_forecast/analysis/nf_artifact_analysis.py` の Ruff format 差分を修正

## ユーザー環境での確認

```bash
cd /mnt/e/env/fc/loto_forecast_project
export PYTHONPATH=src
export UV_TORCH_BACKEND=cpu

uv run python -m compileall -q src tests tools evals scripts
uv run ruff check src tests --no-fix
uv run ruff format src tests --check
uv run mypy src/loto_forecast --ignore-missing-imports
uv run bandit -r src/loto_forecast -c pyproject.toml
```
