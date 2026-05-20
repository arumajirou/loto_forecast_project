# v7 Observability Repair Report

## 結論

v7では、ブラウザ操作・スクリーンショット・console/networkログ・trace・アプリ内イベント・重複検知・エラー早期検知を、DB非依存のローカル観測基盤として追加した。

## 追加・変更

- `src/loto_forecast/observability/store.py`
  - JSONLイベントストア
  - 秘密情報マスク
  - fingerprintによる重複検知
  - Traceback/Exception/timeout/failed 系のエラー早期検知
  - ブラウザRun・スクリーンショット・trace集計
- `scripts/collect_browser_observability.py`
  - Playwrightによる段階的ブラウザ探索
  - safe-clicks既定で危険ボタンをスキップ
  - screenshots / console / network / page_errors / HAR / trace を保存
- `scripts/run_dashboard_observability.sh`
  - dashboard起動
  - HTTP readiness待ち
  - ブラウザ収集
  - summary作成
- `operations_dashboard.py`
  - `観測・診断` パネル追加
  - 概要、エラー早期検知、重複検知、ブラウザ収集、イベントログ、運用コマンドを可視化
- `pyproject.toml`
  - `browser` / `observability` extras追加

## 保存先

```text
artifacts/observability/
  events.jsonl
  browser_runs/<run_id>/
    screenshots/
    console.jsonl
    network.jsonl
    page_errors.jsonl
    visited.jsonl
    manifest.json
    trace.zip
    network.har
  reports/
```

## 実行済み検査

```bash
PYTHONPATH=src python -m compileall -q src tests tools evals scripts
python -m pytest tests/unit/test_observability_store.py -q --no-cov
```

結果: PASS

## 未実行

- Playwright実ブラウザ収集
- Streamlit UIの実起動
- DB接続
- db-init実適用
- dataset書き込み
- 学習・grid-run・E2E長時間実行

## 安全制約

既定では `safe-clicks` により、`db-init`、削除、初期化、実行、書き込みなどの文言を含む操作をクリックしない。
