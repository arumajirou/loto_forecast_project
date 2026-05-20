# 17. NeuralForecast リソース解析 DBスキーマ設計

## 1. 論理ER
- `resources.run (run_id PK)`
  - 1:N `resources.stage_span (run_id FK)`
  - 1:N `resources.resource_metric (run_id FK)`
  - 1:N `log.run_history (run_id)`
  - 1:N `log.error_event (run_id)`

## 2. 設計方針
- 新規テーブル追加なし。
- 既存スキーマの横断JOINで要件を満たす。
- joinキーは `run_id::text` に統一。

## 3. 推奨インデックス
- `resources.run(run_id)`
- `resources.run(started_at DESC)`
- `resources.stage_span(run_id, started_at)`
- `resources.resource_metric(run_id, sampled_at)`
- `log.run_history(run_id, event_ts DESC)`
- `log.error_event(run_id, event_ts DESC)`
