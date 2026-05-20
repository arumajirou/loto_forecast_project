# Observability v7

## 結論

v7では、ブラウザ操作・スクリーンショット・console/networkログ・trace・アプリ内イベント・重複検知・エラー早期検知を、DB非依存のローカル観測基盤として追加した。

## 追加コンポーネント

| パス | 役割 |
|---|---|
| `src/loto_forecast/observability/store.py` | JSONLイベントストア、重複検知、エラー分類、サマリ生成 |
| `scripts/collect_browser_observability.py` | Playwrightによるスクリーンショット、console、network、trace収集 |
| `scripts/run_dashboard_observability.sh` | dashboard起動、疎通待ち、ブラウザ収集、サマリ生成の一括実行 |
| `scripts/observability_summary.py` | 収集済みイベント・成果物のJSON要約 |
| `operations_dashboard.py` の `観測・診断` パネル | 収集結果の可視化、重複/エラー早期検知UI |

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

## 安全方針

- 既定のブラウザ収集は `safe-clicks`。
- `db-init`、削除、初期化、実行、書き込みなどの文言を含むボタンはクリックしない。
- DB書き込み、dataset書き込み、学習実行、E2Eの長時間実行はユーザー承認制。
- 秘密情報らしき値はイベント保存前にマスクする。

## 実行例

```bash
export UV_LINK_MODE=copy
LOTO_UV_ENV_MODE=browser ./scripts/setup_uv.sh
uv run --no-sync playwright install chromium
./scripts/run_dashboard_observability.sh --max-clicks 40 --max-depth 2
```
