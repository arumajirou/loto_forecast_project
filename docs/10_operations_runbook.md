# 運用手順書（詳細）

## 1. 初期セットアップ

1. `.env` 作成
2. 依存インストール
3. `python -m loto_forecast.cli db-init`
4. `python -m loto_forecast.cli catalog-import --library neuralforecast`

## 2. 日次運用

1. `train`
2. `predict`
3. `evaluate`
4. 閾値超過なら `retrain`

## 2.1 外生変数ETL（日次または再学習前）

```bash
python -m loto_forecast.cli build-exog \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table loto_y_ts_exog \
  --group-cols 'loto,unique_id,ts_type' \
  --parallel-workers 4 --enable-gpu-compute
```

- 接頭辞規約: `hist_`, `stat_`, `feat_`
- 実行メトリクス保存先: `resources.run`, `resources.stage_span`, `resources.resource_metric`, `resources.metric_def`

## 2.2 UNI2TS埋め込み外生変数ETL（日次または再学習前）

```bash
python -m loto_forecast.cli build-exog-uni2ts \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table uni2ts \
  --group-cols 'loto,unique_id,ts_type' \
  --context-length 128 --embedding-dim 256 \
  --parallel-workers 4 --enable-gpu-compute
```

- 接頭辞規約: `hist_uni2ts_*`
- 付加列: `embedding_dim`, `model_name`, `model_version`, `config_hash`, `created_at`, `updated_at`, `y_idx`
- 実行メトリクス保存先: `resources.run`, `resources.stage_span`, `resources.resource_metric`, `resources.metric_def`

## 2.3 TimesFM埋め込み外生変数ETL（日次または再学習前）

```bash
python -m loto_forecast.cli build-exog-timesfm \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table timesfm \
  --group-cols 'loto,ts_type' \
  --backend timesfm_forecast_features \
  --window-size 128 --min-points 16 --embedding-dim 256 \
  --parallel-workers 4 --enable-gpu-compute \
  --only-missing --if-exists append
```

- 接頭辞規約: `hist_timesfm_*`
- 付加列: `loto_y_ts_row_id`, `embedding_dim`, `model_name`, `model_version`, `config_hash`, `created_at`, `updated_at`, `y_idx`
- 実行メトリクス保存先: `resources.run`, `resources.stage_span`, `resources.resource_metric`, `resources.metric_def`

## 2.4 Chronos埋め込み外生変数ETL（日次または再学習前）

```bash
python -m loto_forecast.cli build-exog-chronos \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table chronos \
  --group-cols 'loto,unique_id,ts_type' \
  --backend chronos_pipeline_auto \
  --window-size 128 --min-points 16 --embedding-dim 256 --batch-size 256 \
  --parallel-workers 4 --enable-gpu-compute \
  --only-missing --if-exists append
```

- 接頭辞規約: `hist_chronos_*`
- 付加列: `loto_y_ts_row_id`, `embedding_dim`, `model_name`, `model_version`, `config_hash`, `created_at`, `updated_at`, `y_idx`
- 高速化: `zero+zscore` はベクトル化処理、Chronos推論はGPUバッチ処理、特徴量バックエンドは並列グループ処理
- 実行メトリクス保存先: `resources.run`, `resources.stage_span`, `resources.resource_metric`, `resources.metric_def`

## 2.5 統合学習データセット作成（日次または再学習前）

```bash
python -m loto_forecast.cli build-unified-dataset \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --base-schema dataset --base-table loto_y_ts \
  --hist-schema dataset --hist-table loto_hist_feat \
  --exog-schema exog \
  --output-schema dataset --output-table loto_y_ts_unified
```

- 結合元: `dataset.loto_y_ts` + `dataset.loto_hist_feat` + `exog.*`
- 出力: `dataset.loto_y_ts_unified` + `csv/parquet/spark`（設定時）
- 外生区分規約: `stat_`=static, `hist_`=historical, `feat_`=feature(future)

## 2.6 メタテーブル駆動の網羅/再帰実行

```bash
python -m loto_forecast.cli meta-automodel-run --limit 100
```

- 実行定義: `meta.nf_automodel`
- 結果保存: `model.nf_automodel`
- `param_space_json` を展開し網羅実行、`recursive_depth` 回だけ再帰実行
- BaseAuto引数は `auto_*` カラムで管理（`auto_cls_model`, `auto_h`, `auto_config_json`, `auto_loss`, `auto_valid_loss`, `auto_search_alg`, `auto_num_samples`, `auto_backend` など）
- `unified_filter_json` で対象系列を絞り込み可能（例: `{"loto":"bingo5","unique_id":"N1","ts_type":"raw"}`）
- `unified_group_cols_json`（既定: `["loto","unique_id","ts_type"]`）と `unified_group_validate_strict` で、`loto_y_ts_unified` のグループ検証を制御
- モデル保存/ロード/解析は `run_save`, `run_load`, `run_analyze` と `save_*` カラムで制御

## 2.7 PySparkテーブル実行

```bash
python -m loto_forecast.cli run-table-pyspark \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts_unified \
  --source-sql "SELECT * FROM \"dataset\".\"loto_y_ts_unified\" WHERE y IS NOT NULL AND loto = 'bingo5' AND unique_id = 'N1' AND ts_type = 'raw'" \
  --target-schema dataset --target-table loto_y_ts_unified_spark \
  --output-if-exists replace \
  --output-parquet-path ./artifacts/datasets/loto_y_ts_unified_spark.parquet \
  --execution-backend auto --skip-row-count --no-spark-ui-enabled \
  --postgres-write-mode copy --postgres-copy-chunk-rows 50000
```
`source-sql` で条件を先に絞ることで、Spark 側に全件読込せず高速化できます。
`execution-backend` は `auto|polars|dask|pandas|spark` を選択可能です。

## 2.7.1 unified グループ整合性チェック

```bash
python -m loto_forecast.cli check-unified-grouping \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --schema dataset --table loto_y_ts_unified \
  --group-cols loto,unique_id,ts_type \
  --time-col ds
```

## 2.8 高速一括実行（bash）

```bash
bash scripts/run_fast_meta_pipeline.sh
```

- 内部実行: `db-init` -> `meta-automodel-create` -> `build-unified-dataset --fast-mode --postgres-write-mode copy` -> `meta-automodel-run` -> `run-table-pyspark`
- `META_LIMIT` などの環境変数で調整可能

## 2.8.1 ローカル学習フル一括（bash）

```bash
bash scripts/run_local_nf_full_pipeline.sh
```

- 内部実行: `run_local_nf_meta_create_and_pyspark.sh` -> `run_model_save_load_analyze.sh`
- 目的: 学習/予測/評価に加えて、モデル保存・ロード・解析を1コマンドで実行

## 2.9 Streamlit Runner運用

`streamlit run scripts/operations_dashboard.py` の `Operations -> Runner` から以下を実行:
- `meta-automodel-create`
- `run-table-pyspark`
- `meta-automodel-run`
- `model-save-load-analyze`
- `run_local_nf_full_pipeline.sh`（フル一括）
- Fast Sequence（高速プリセット）

実行時は進捗バー、ステージ表示、経過時間、stdout/stderr tail、最終ログ時刻を表示し、完了後に return code と時刻を確認できます。
`Status. Live Meta/Model Status` で `meta.nf_automodel` / `model.nf_automodel` の最新状態を確認できます。
CLIバッチ（`scripts/run_local_nf_meta_create_and_pyspark.sh`）では stepごとに `START/DONE/FAIL`、経過秒、ログファイルを表示します。

## 2.10 モデル保存/ロード/解析（単体実行）

```bash
python -m loto_forecast.cli model-save-load-analyze \
  --run-id <RUN_ID> \
  --source-path ./artifacts/<RUN_ID> \
  --save-path ./artifacts/saved_models/<RUN_ID> \
  --run-save --run-load --run-analyze \
  --save-dataset --save-overwrite
```

- bash実行: `bash scripts/run_model_save_load_analyze.sh`
- `RUN_ID` 未指定時は `CONFIG_NAME` / `CONFIG_ID` で最新runを自動解決可能（`AUTO_RESOLVE_RUN_ID=true`）
- 解析結果は `model.nf_automodel.model_save_json/model_load_json/model_analyze_json` に保存される

## 3. 週次/検証運用

1. `grid-create`
2. `grid-run`
3. `grid-status`
4. 上位設定を採用し本番相当 run 実施

## 4. 障害対応

- DB接続失敗: `.env` と PostgreSQL 稼働確認
- `db-init` 失敗: 最新コード反映後に再実行（`python -m loto_forecast.cli db-init`）
- 必須列不足: `unique_id/ds/y` を確認
- 外生不足: futr生成可能な列へ分離するかモデル設定変更
- 失敗run調査: `logs/*.log`, `model_run.error_message`, `execution_event_log`
- `meta-automodel-run` 失敗: `meta.nf_automodel` / `model.nf_automodel` の存在確認
- `run-table-pyspark` が `TypeError: 'JavaPackage' object is not callable` の場合:
  - `SPARK_HOME` と `pyspark` のミスマッチが主因。`echo $SPARK_HOME` を確認。
  - `unset SPARK_HOME` 後に再実行する。
  - 本実装では `SPARK_HOME` 無効化で再試行し、失敗時は `pandas` フォールバックに切替える。
- `run-table-pyspark` が `ClassNotFoundException: org.postgresql.Driver` の場合:
  - Spark JDBCドライバ未配置が主因。
  - 本実装では Spark 実行中エラーでも `pandas` フォールバックに切替えるため、JSON の `fallback_reason` を確認する。
- `run-table-pyspark` が `polars read done ...` の後で進まない場合:
  - 旧runの停止ジョブ/DBロック待ちで `replace` が待機している可能性が高い。
  - 同一シェルで `jobs -l` を実行し、停止ジョブがあれば `kill %<job-id>` で終了する。
  - `POSTGRES_LOCK_TIMEOUT_MS`（既定10秒）を使って無限待機を避ける。
  - 進捗は `tail -f logs/pipeline_runs/*_run-table-pyspark.log` で確認する。
- `run_meta_automodel_create.sh` で JSONDecodeError の場合:
  - `MODEL_PARAMS_JSON` / `PARAM_SPACE_JSON` を必ず JSON 文字列で渡す（ダブルクォートを含める）。
  - 例: `MODEL_PARAMS_JSON='{"backend":"optuna","num_samples":20}'`
- `model-save-load-analyze` で load失敗する場合:
  - `--source-path` / `--save-path` のディレクトリに `.ckpt/.pkl` が存在するか確認
  - `save_dataset=False` かつ `--load-check-predict` の場合は `predict_insample` が失敗することがあるため、`--no-load-check-predict` を指定

## 5. ログ保全

- ログファイルは `logs/` に保存
- DB側は `execution_event_log` と `grid_search_task.error_message` を保全
