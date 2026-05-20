# V8 Startup Repair Notes

## 結論

v8 は、v7 の静的検査成功後に残った dashboard 非表示問題を対象にした起動安定化パッチです。

## 修正内容

- `resources.__init__` の eager import を廃止し、optional pipeline を lazy import 化。
- `from resources.utils import ...` が Chronos/TimesFM/Uni2TS/DB writer を連鎖 import しないよう修正。
- `scripts/run_dashboard_observability.sh` が既存 `.venv` を毎回削除しないよう修正。
- readiness timeout を既定 180 秒へ拡張。
- readiness 失敗時に Streamlit ログ末尾と process 情報を必ず表示。
- `run_operations_dashboard.sh` が dashboard 依存不足を検出した場合だけ `uv sync` するよう修正。
- `scripts/diagnose_dashboard_startup.sh` と `scripts/check_dashboard_import.py` を追加。

## 未実行

- DB 書き込み
- `db-init` 実適用
- dataset 書き込み
- 学習
- 長時間 E2E
