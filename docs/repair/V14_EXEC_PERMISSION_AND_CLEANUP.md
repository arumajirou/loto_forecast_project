# V14: execute-bit loss and stopped Playwright cleanup

## 結論
ZIP展開や `/mnt/e` 上のファイルコピーで shell script の実行権限が落ちても動くように、
内部スクリプト呼び出しを `bash ./scripts/<name>.sh` に変更しました。

## 修正内容
- `scripts/start_dashboard_and_capture_screenshots.sh`
  - `./scripts/wsl_start_loto_app.sh` を `bash ./scripts/wsl_start_loto_app.sh` に変更
  - `./scripts/capture_app_screenshots.sh` を `bash ./scripts/capture_app_screenshots.sh` に変更
- `scripts/wsl_start_loto_app.sh`
  - setup script 呼び出しを `bash ./scripts/setup_uv.sh` に変更
- `scripts/capture_app_screenshots.sh`
  - setup script 呼び出しを `bash ./scripts/setup_uv.sh` に変更
- `scripts/kill_loto_observability_processes.sh`
  - 停止中 (`Tl`) の Playwright / Chromium / dashboard を `CONT -> TERM -> KILL` の順で掃除

## 安全
DB接続、DB書き込み、dataset書き込み、db-init実適用、学習、E2Eは含みません。
