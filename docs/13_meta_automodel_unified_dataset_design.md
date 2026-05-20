# Meta AutoModel / Unified Dataset Design

## 1. 目的
- `dataset.loto_y_ts` を基準に `dataset.loto_hist_feat` と `exog.*` 全テーブルを統合した学習データセットを作成する。
- `meta.nf_automodel` に定義したパラメータ/データセット設定から網羅的に AutoModel 実行する。
- モデル成果物・評価・解析結果を `model.nf_automodel` に保存する。
- `NeuralForecast.save/load` を利用したモデル保存/ロード/解析を実行し、結果を回収する。
- 受け渡し先として `psql` / `csv` / `pyspark(parquet/csv)` を同時出力する。

## 2. 追加ディレクトリと責務
- `src/loto_forecast/data/unified_dataset.py`
  - 統合データセット生成
  - JOINキー自動推定
  - PostgreSQL / CSV / Parquet / Spark出力
- `src/loto_forecast/orchestration/meta_automodel.py`
  - `meta.nf_automodel` 読み込み
  - `param_space_json` 展開
  - 再帰ラウンド実行 (`recursive_depth`)
  - `model.nf_automodel` 保存
- `sql/03_create_nf_automodel_tables.sql`
  - `meta/model` スキーマと `nf_automodel` テーブル作成

## 3. JOIN戦略
- 基準: `dataset.loto_y_ts`
- 優先JOINキー:
  1. `loto_y_ts_row_id`
  2. `row_id`
  3. `unique_id` + `ds` (+ `loto`, `ts_type` があれば追加)
- 衝突列名は接頭辞を維持して改名
  - `hist_*` -> `hist_<table>_*`
  - `stat_*` -> `stat_<table>_*`
  - `feat_*` -> `feat_<table>_*`

## 4. 外生変数区分
- `stat_*` -> `stat_exog`
- `hist_*` -> `hist_exog`
- `feat_*` -> `futr_exog`
- 既存互換として、カレンダー列は `futr_exog`、未分類列は `hist_exog` 扱い。

## 5. メタ実行フロー
1. `meta.nf_automodel` から `active=true` 設定を取得。
2. 統合データセットを生成して `dataset.<output_table>` に保存。
3. `param_space_json` を直積展開し網羅実行。
4. `recursive_depth` 回だけ seed をずらして再帰実行。
5. 学習/予測/評価/Permutation重要度を実行。
6. 必要に応じて `save/load/analyze` を実行。
7. 結果を `model.nf_automodel` へ upsert。
8. `meta.nf_automodel.last_*` を更新。

## 6. 主要テーブル
- `meta.nf_automodel`: 実行定義（モデル、BaseAuto引数 `auto_*`、保存/ロード/解析フラグ、データセット設定）
  - BaseAuto対応カラム例: `auto_cls_model`, `auto_h`, `auto_config_json`, `auto_loss`, `auto_valid_loss`, `auto_search_alg`, `auto_num_samples`, `auto_cpus`, `auto_gpus`, `auto_backend`
  - データ絞り込みカラム例: `unified_filter_json`（例: `{"loto":"bingo5","unique_id":"N1","ts_type":"raw"}`）
  - グループ検証カラム例: `unified_group_cols_json`, `unified_group_validate_strict`
- `model.nf_automodel`: 実行結果（status, metrics_json, diagnostics_json, explain_json, `model_save_json`, `model_load_json`, `model_analyze_json`, artifact_path）

## 7. CLI
- 統合データセット作成:
  - `python -m loto_forecast.cli build-unified-dataset ...`
- メタ駆動実行:
  - `python -m loto_forecast.cli meta-automodel-run --limit 100`

## 8. Streamlit連携
- `scripts/operations_dashboard.py` の `Operations -> Model/Grid/Meta` で以下を確認可能:
  - `dataset.model_run`, `dataset.grid_search_*`, `dataset.execution_event_log`
  - `meta.nf_automodel`, `model.nf_automodel`
  - `dataset.loto_y_ts_unified` サンプル
- `Operations -> Runner` に以下を追加:
  - `meta-automodel-create` 実行
  - `run-table-pyspark` 実行
  - `model-save-load-analyze` 実行
  - `db-init -> unified(fast) -> meta-run -> pyspark` 高速一括実行
- 実行中は進捗バー、ステージ表示、経過時間、stdout/stderr tail、最終ログ時刻を表示。
- `Command Lab` に unified/meta/pyspark 実行のクイックコマンドを追加し、実行時にも進捗を表示。
