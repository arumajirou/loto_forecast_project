# DB・スキーマ・テーブル設計書

## スキーマ

- `dataset`

## 1. 実行管理テーブル

- `model_run`: run単位メタ（library/adapter/status/grid連携含む）
- `model_metric`: run×metric
- `forecast`: run×series×timestamp の予測値
- `exog_contribution`: 外生寄与
- `resource_sample`: CPU/Memory/RSS サンプル

## 1.1 外生特徴量テーブル

- `exog.loto_y_ts_exog`（デフォルト）
  - 主なキー: `loto`, `unique_id`, `ts_type`, `ds`
  - 目的変数: `y`
  - 外生特徴:
    - `hist_*`（履歴系）
    - `stat_*`（静的統計）
    - `feat_*`（時刻/補助特徴）

## 2. codegen カタログテーブル

- `library_catalog`
- `module_catalog`
- `symbol_catalog`
- `symbol_param_catalog`

用途:
- API仕様の正規化保管
- 引数漏れや未知引数を事前検知

## 3. グリッド実行テーブル

- `grid_search_definition`: グリッド定義（param_space, horizon, adapter）
- `grid_search_task`: 展開済みタスクと状態遷移
- `execution_event_log`: 実行イベント（start/success/fail）

## 4. リソース監視テーブル（resources schema）

- `resources.metric_def`
- `resources.resource_metric`
- `resources.run`
- `resources.stage_span`

## 4. DDL配置

- `sql/00_create_schema.sql`
- `sql/01_create_meta_tables.sql`
- `sql/02_create_catalog_and_grid_tables.sql`
