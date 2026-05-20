# UI Redesign Notification Plan

## 要件定義サマリ
- 現状の NeuralForecast 実行・検証ラボは、設定、実行、検証、保存が単一関数に密結合し、初心者が必須入力と自動補完の境界を把握しづらい。
- 「有効な総当たり組合せ数 0」の原因が技術ロジック寄りで、原因、影響、対処が分離されていない。
- 実行開始、成功、失敗、例外、長時間処理完了に対して通知経路がなく、操作の確定感と次の推奨行動が不足している。
- 通知失敗で本処理が落ちてはいけない。SMTP 未設定、音声再生不可、外部 provider 未設定でも UI は継続動作する必要がある。

## 基本設計サマリ
- 画面導線は `かんたん / 標準 / 詳細` の 3 モードに再編し、学習系は step wizard を先頭に置く。
- NotificationService は application 層に置き、screen / beep / email / mock message の adapter を差し込む。
- 主要操作はまず `action_confirmed` を発火し、その後 `operation_start`, `operation_success|failure`, `long_running_*`, `exception` を一貫して通す。
- 画面通知は modal 相当の中央ダイアログ表示と toast の二層構成とし、必ず次の推奨操作を添える。

## 詳細設計サマリ
- `src/loto_forecast/application/notification_events.py`
  - イベント種別、重大度、チャネル、dedup key を定義。
- `src/loto_forecast/application/notification_service.py`
  - dedup、rate limit、監査ログを担当。
- `src/loto_forecast/infrastructure/notifications/*.py`
  - Email, Beep, MockMessage 各 adapter。
- `src/loto_forecast/api/streamlit/ui/*.py`
  - wizard 状態計算、プリセット適用、通知表示、dialog 表示を分離。
- `src/loto_forecast/api/streamlit/operations_dashboard.py`
  - 新 wizard/notification レイヤを統合し、既存の CLI 実行導線を維持したまま UX を改善。
  - 追加で `Count rows` / `Run SQL` / `Directory Compiler` / `Markdown Compiler` / `Async XAI API` / `runtime-model-load` 検証導線も NotificationService に接続。

## 非機能要件
- 通知失敗は warning 扱いとし、本処理の return code を汚染しない。
- SMTP 未設定時は dry-run として記録し、UI に未設定と表示する。
- 音通知が不可能な環境では画面通知のみで完結する。
- テストでは NotificationService を adapter 単位で mock 可能にする。

## 2026-03-31 追加実装
- Playwright 環境
  - ルート直下に `package.json` / `package-lock.json` を追加し、`@playwright/test` を固定
  - システム Chrome (`/usr/bin/google-chrome`) を利用し、追加 browser download は実施していない
  - 実行コマンド
    - `BASE_URL=http://127.0.0.1:8510 ROUTE_NAME=db_connected node tests/e2e/operations_dashboard_redesign_ui_check.mjs`
    - `BASE_URL=http://127.0.0.1:8511 ROUTE_NAME=linux_fallback node tests/e2e/operations_dashboard_redesign_ui_check.mjs`
- 実ブラウザ E2E
  - `db_connected` と `linux_fallback` を `artifacts/logs/browser_observation_detailed.json` に別 route として保存
  - スクリーンショットを `artifacts/screenshots/exhaustive/` に保存
  - 確認項目
    - 初期表示
    - Step Wizard 表示
    - `かんたん / 標準 / 詳細` モード切替
    - `最短で試す`
    - `おすすめ設定を自動入力`
    - 0件診断パネル
    - 実行前チェック
    - 通知設定表示
    - エラー時の画面誘導
- メール通知確認
  - `LF_SMTP_HOST` / `LF_NOTIFY_FROM` 未設定のため、`zakumagahiyakesita@gmail.com` 宛ての dry-run のみ確認
  - `OPERATION_START` / `OPERATION_SUCCESS` / `OPERATION_FAILURE` / `LONG_RUNNING_COMPLETE` / `EXCEPTION` の 5 イベントで件名と本文必須項目を検証
- 未解決事項
  - 実 SMTP 送信は環境変数設定後に再検証が必要
  - shell 側の DB クライアント接続が失敗したため、`log.ui_state_snapshot` の before/after 監査は今回更新できていない
