# 詳細テーブル設計書

## 1. `dataset.symbol_catalog`

主キー: `symbol_id`

主要列:
- `library_name`, `module_name`, `symbol_type`, `symbol_name`, `full_path`
- `parent_symbol`, `role`, `return_type`, `event_like`, `docstring`
- `raw` (jsonb: 取込元全情報)

制約:
- `UNIQUE(library_name, full_path)`

## 2. `dataset.symbol_param_catalog`

主キー: `(symbol_id, ordinal)`

主要列:
- `param_name`, `param_kind`, `annotation`
- `has_default`, `default_repr`, `is_required`

## 3. `dataset.grid_search_task`

主キー: `task_id`

主要列:
- `grid_id`, `task_order`, `param_values`, `status`
- `run_id`, `log_path`, `started_at`, `ended_at`
- `result`, `metrics`, `resource_summary`, `error_message`

状態遷移:
- `pending -> running -> success|failed`

## 4. `dataset.execution_event_log`

主要列:
- `task_id`, `run_id`, `event_ts`
- `level`, `event_type`, `message`, `payload`

用途:
- 運用監査、失敗時トレース、時間軸分析
