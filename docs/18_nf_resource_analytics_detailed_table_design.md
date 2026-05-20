# 18. NeuralForecast リソース解析 詳細テーブル設計

## 1. resources.run
- 主利用列:
  - `run_id`, `started_at`, `ended_at`, `status`
  - `app_name`, `command`, `rows_target`, `rows_written`, `rows_failed`
  - `tags`, `error_summary`
- 派生:
  - `duration_sec`, `execution_os`

## 2. resources.stage_span
- 主利用列:
  - `run_id`, `stage_name`, `started_at`, `ended_at`, `duration_ms`
  - `rows_in`, `rows_out`, `db_time_ms`, `db_rows`
  - `gpu_util_avg`, `gpu_mem_used_mb_avg`
  - `exception_type`, `exception_msg`

## 3. resources.resource_metric
- 主利用列:
  - `run_id`, `sampled_at`, `metric_key`, `metric_value`, `unit`
- 用途:
  - run選択時の波形可視化、異常時の裏取り

## 4. log.run_history
- 主利用列:
  - `run_id`, `event_ts`, `event_type`, `status`
  - `model_name`, `dataset_name`, `message`
- 用途:
  - runタイムライン、エラー直前イベント特定

## 5. log.error_event
- 主利用列:
  - `run_id`, `event_ts`, `model_name`, `stage`
  - `error_type`, `error_message`
- 用途:
  - エラー分類、頻度、再現候補の切り分け
