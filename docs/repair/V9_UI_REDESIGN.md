# V9 UI/UX Redesign Report

## 結論
NeuralForecast 実行・検証ラボを、DB未接続でも使える `NeuralForecast Cockpit` 中心の画面へ再設計しました。

## 背景
ユーザー環境では dashboard は表示されましたが、DB未接続時に `DB未接続のため表示できません。` が前面に出て、NeuralForecast 実行・検証ラボの導線が分かりにくい状態でした。

## 変更内容
- `NeuralForecast Cockpit` を表示パネルの先頭に追加
- 既存の `NeuralForecast 実行・検証ラボ` は `NeuralForecast 詳細ラボ` として後方互換に分離
- DB未接続時も Cockpit を表示
- DB接続失敗を大きなエラー固定表示から、折りたたみ可能な診断表示へ変更
- DB_PASSWORD 未設定を早期検知し、`.env` / 環境変数の設定手順を画面に表示
- 実行計画レビュー画面を追加
- 観測イベント、ログ、Traceback、DB/authエラー、timeoutを集約表示
- 画面/機能の重複候補を棚卸しして、責務分離方針を表示

## 主な追加ファイル
- `src/loto_forecast/api/streamlit/nf_lab_cockpit.py`
- `docs/repair/V9_UI_REDESIGN.md`
- `docs/repair/V9_SETUP_AND_VERIFY_COMMANDS.md`
- `docs/repair/V9_CHANGE_MANIFEST.json`

## 変更ファイル
- `src/loto_forecast/api/streamlit/operations_dashboard.py`

## 実行済み確認
```bash
PYTHONPATH=src python -m compileall -q src tests tools evals scripts
PYTHONPATH=src python -m pytest tests/unit/test_observability_store.py -q --no-cov
```

## 未実行
- `uv sync`
- `ruff`
- `mypy`
- `bandit`
- Streamlit実ブラウザ起動
- DB接続
- `db-init` 実適用
- 学習、E2E、grid-run

## 安全制約
- DB書き込みなし
- dataset書き込みなし
- db-init実適用なし
- `--password` 引数なし
- 学習/長時間ジョブなし
