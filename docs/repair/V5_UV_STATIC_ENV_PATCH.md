# loto_forecast_project uv v5 修正レポート

## 結論

v4で発生した `uv sync --extra dev --locked` による CUDA / nvidia wheel ダウンロード失敗を避けるため、依存関係を「静的検査用の軽量環境」と「dashboard / 学習用のフル環境」に分離した。

## 原因

- 配布ZIPに含まれた `uv.lock` が、生成元環境またはミラーに依存した wheel URL を持っていた。
- `--locked` により、その lock が固定され、ローカル環境で到達不能な URL を取得しようとした。
- `UV_TORCH_BACKEND=cpu` は `uv pip` では有効だが、project-mode の `uv sync` では pyproject 側の sources/index 設計または lock の再生成が必要になる。
- v4では `dev` 同期でも base dependency の `neuralforecast` が入り、torch/CUDA 依存を引き込んでいた。

## 変更内容

- `uv.lock` を配布ZIPから削除。
- `pyproject.toml` の base dependencies を軽量化。
- ML / dashboard 依存を optional extra に分離。
  - `dev`: ruff, mypy, pytest, bandit など
  - `dashboard`: streamlit, plotly, fastapi など
  - `ml`: neuralforecast, statsmodels, ray, shap など
  - `full`: dashboard + ml 相当
- `scripts/setup_uv.sh` を static / dashboard / full モード対応へ変更。
- `scripts/verify_static.sh` は `uv sync --extra dev` のみを使う。
- `scripts/fix_style.sh` / `scripts/repair_static.sh` も軽量 dev 環境のみを使う。
- `Makefile` から `--locked` 既定を削除。
- README の uv 手順を修正。

## 実行済み

```bash
PYTHONPATH=src python -m compileall -q src tests tools evals scripts
```

結果: PASS

## 未実行

- `uv sync --extra dev`
- `ruff`
- `mypy`
- `bandit`
- pytest
- dashboard起動
- DB接続
- DB書き込み
- 学習/E2E

## 安全制約

- `db-init` 実適用は未実行。
- `dataset` 書き込みは未実行。
- 外部送信、学習、ブラウザE2Eは未実行。
