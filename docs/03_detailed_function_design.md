# 詳細機能設計書

## 1. `catalog/codegen_catalog.py`

- `load_codegen_yaml(path)`
  - YAML読込、ルート構造検証
- `parse_codegen_rows(payload)`
  - 行を `SymbolRecord` に正規化
  - `method/property` の `parent_symbol` 推定
- `upsert_codegen_catalog(engine, yaml_path, library_name)`
  - `library_catalog/module_catalog/symbol_catalog/symbol_param_catalog` へ投入
- `validate_call_arguments(engine, library_name, full_path, arguments)`
  - 必須漏れ・未知引数・型ヒント整合性を返却

## 2. `orchestration/grid_runner.py`

- `expand_param_grid(param_space, max_tasks)`
  - パラメータ空間を再帰的に全組合せ展開
- `create_grid(...)`
  - definition 登録 + task展開
- `run_grid(grid_id)`
  - pending task を順次実行
  - 状態遷移、結果、メトリクス、資源、イベントログを保存

## 3. `models/registry.py`

- `NeuralForecastAutoAdapter`
  - モデル一覧
  - `inspect.signature` に基づくパラメータ妥当性検証
  - `train/predict/evaluate` の共通実行

## 4. `orchestration/pipeline.py`

- `prepare_dataset(...)`
  - time/cyclical/lag/rolling/diff 生成
- `train(...)`
  - DB読込 -> 学習 -> model_run登録
- `retrain(base_run_id)`
  - 過去metaを起点に再学習
- `predict(run_id)`
  - futr_df作成、予測保存
- `evaluate(run_id)`
  - holdout評価 + 検定 + metric保存

## 5. `infra/meta_store.py`

- `upsert_model_run`, `mark_model_run_end`
- `create_grid_definition`, `replace_grid_tasks`, `start_grid_task`, `finish_grid_task`
- `log_execution_event`, `write_resource_samples`

## 6. `orchestration/meta_automodel.py`

- `create_meta_automodel_config(config, upsert_by_name=True)`
  - `meta.nf_automodel` の設定行を作成/更新
  - `config_name` キーでupsert可能
  - `model_params_json` / `param_space_json` / `auto_callbacks_json` / `auto_config_json` の正規化
  - BaseAuto引数（`auto_*`）と save/load/analyze 実行設定を保持
- `run_meta_automodel(config_id=None, limit=100, stop_on_error=False)`
  - `param_space_json` の網羅展開 + `recursive_depth` 再帰実行
  - `NeuralForecast.save/load` とモデル保存物解析を実行
  - `model.nf_automodel` に `model_save_json/model_load_json/model_analyze_json` を保存

## 7. `data/spark_table_runner.py`

- `run_table_with_pyspark(spec)`
  - PostgreSQL(JDBC) からテーブル/SQLを入力
  - Spark SQL (`transform_sql`) で加工
  - PostgreSQL / Parquet / CSV / Sparkデータセットへ出力
  - `replace/append/fail` モードを統一制御
