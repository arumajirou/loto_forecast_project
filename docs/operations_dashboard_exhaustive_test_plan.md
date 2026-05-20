# Operations Dashboard Exhaustive Test Plan

## Goal
- Fill coverage for major dashboard panels and representative normal/error/boundary flows.
- Keep Linux fallback and connected evidence separate.
- Record DB and file side effects with before/after observations.

## Routes
- `8511` fallback route: `DB_HOST=127.0.0.2`
- `8510` connected route: `DB_PASSWORD=<set-in-env>`
- Windows Chrome: reference-only in `artifacts/logs/browser_runtime_identity.md`

## Covered Flows
- Fallback home and `運用` degraded tabs
- `ディレクトリ統合` invalid input
- `ディレクトリ統合` compile path
- `Markdown統合` compile path
- `成果物・ログ` event log viewing
- Connected `概要`
- Connected `NeuralForecast 実行・検証ラボ` train and save/load navigation
- Connected `リソース分析`
- Connected `スキーマ出力`
- AppTest, unit test, and E2E smoke

## Exclusions / Reasons
- `db-init`, `schema:log テーブル初期化`, destructive DB admin actions, and long-running runner commands were not executed because they are destructive or high-cost under AGENTS.md.
- Windows Chrome was not re-driven in this pass to avoid mixing prior font/runtime evidence with the current functional Linux audit.
