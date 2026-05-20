# V15 bash invocation fix

## 結論

`start_dashboard_and_capture_screenshots.sh` 内の誤った `bash bash ./scripts/wsl_start_loto_app.sh`
呼び出しを `bash ./scripts/wsl_start_loto_app.sh` に修正した。

## 修正理由

WSLで `/usr/bin/bash: /usr/bin/bash: バイナリファイルを実行できません` が発生した。
これは `bash` をスクリプトファイルとして二重指定していたため。

## 追加修正

`kill_loto_observability_processes.sh` は、停止中の Playwright/node プロセスへ `CONT` を送ってから終了させると
EPIPEスタックトレースを出すことがあるため、停止プロセスは直接 `KILL` する方式に変更した。

## 未実行

DB書き込み、dataset書き込み、db-init、学習、E2E、ブラウザ実収集は未実行。
