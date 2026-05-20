# UPLOAD_PACKAGE_V17 — 解析資料アップロード用ZIP作成

## 目的

`artifacts/observability`、スクリーンショット、HAR、trace、manifest、進捗ログ、`docs/repair` の改修レポートを、外部レビューやChatGPTへの再アップロードに使える1つのZIPへまとめます。

## 実行

```bash
cd /mnt/e/env/fc/loto_forecast_project

bash ./scripts/package_analysis_upload.sh --tree --note "v16 browser observability run"
```

出力先:

```text
artifacts/upload_packages/loto_analysis_upload_package_<UTC>.zip
artifacts/upload_packages/latest_upload_package.zip
```

## 含まれるもの

- `UPLOAD_PACKAGE_README.md`
- `summary.json`
- `manifest_files.json`
- `project_diagnostics/`
- `project_files/docs/repair/`
- `project_files/artifacts/observability/browser_runs/`
- screenshots
- `network.har`
- `trace.zip`
- `console.jsonl`
- `progress.jsonl`
- `visited.jsonl`

## 除外するもの

- `.venv`
- `node_modules`
- `.git`
- キャッシュ
- Python bytecode
- Windows `:Zone.Identifier`
- 既定で50MBを超えるファイル

## オプション

```bash
bash ./scripts/package_analysis_upload.sh --no-traces
bash ./scripts/package_analysis_upload.sh --no-har
bash ./scripts/package_analysis_upload.sh --no-screenshots
bash ./scripts/package_analysis_upload.sh --max-file-mb 20
bash ./scripts/package_analysis_upload.sh --git-diff --tree
```

## 安全

このスクリプトはDB接続、DB書き込み、学習、`db-init`、cron導入を実行しません。
コマンド出力に含まれる代表的な password / token / DB URL は簡易マスクします。
