# 各種コマンド

> 注: 複数行コマンドを貼る場合は行末に `\` を入れてください。
> 例: `--database loto \` のようにしないと次行が別コマンドとして実行されます。

## DB初期化

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli db-init
```

## カタログ取り込み・検証

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli catalog-import --library neuralforecast --yaml-path ./docs/lib_docs/neuralforecast_all_codegen.yaml
PYTHONPATH=src uv run python -m loto_forecast.cli catalog-list --library neuralforecast --limit 20
PYTHONPATH=src uv run python -m loto_forecast.cli catalog-validate --library neuralforecast --full-path neuralforecast.auto.AutoNHITS.__init__ --arguments-json '{"h":28,"num_samples":10}'
```

## 学習/再学習/予測/評価

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli train --model AutoNHITS --h 28 --params-json '{"num_samples":10,"seed":1}'
PYTHONPATH=src uv run python -m loto_forecast.cli retrain --base-run-id <RUN_ID>
PYTHONPATH=src uv run python -m loto_forecast.cli predict --run-id <RUN_ID> --h 28
PYTHONPATH=src uv run python -m loto_forecast.cli evaluate --run-id <RUN_ID>
PYTHONPATH=src uv run python -m loto_forecast.cli explain --run-id <RUN_ID> --method permutation
PYTHONPATH=src uv run python -m loto_forecast.cli explain --run-id <RUN_ID> --method granger --maxlag 8 --top-k 20
```

## グリッドサーチ

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli grid-create --grid-id nf_grid_001 --adapter neuralforecast_auto --model AutoNHITS --h 28 --param-space-json '{"num_samples":[10,20],"seed":[1,2],"backend":["optuna"]}'
PYTHONPATH=src uv run python -m loto_forecast.cli grid-run --grid-id nf_grid_001
PYTHONPATH=src uv run python -m loto_forecast.cli grid-status --grid-id nf_grid_001
```

## アダプタ確認

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli adapters
```

## 外生変数生成（exog）+ リソース計測（resources）

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-exog \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table loto_y_ts_exog \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y \
  --parallel-workers 4 --enable-gpu-compute --sampling-interval-sec 1.0 \
  --lib-docs-dir ./docs/lib_docs
```

## UNI2TS埋め込み外生変数生成（exog.uni2ts）+ リソース計測（resources）

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-exog-uni2ts \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table uni2ts \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y \
  --context-length 128 --embedding-dim 256 --batch-size 512 \
  --parallel-workers 4 --enable-gpu-compute --sampling-interval-sec 1.0 \
  --uni2ts-codegen-yaml ./docs/lib_docs/uni2ts_all_codegen.yaml
```

## TimesFM埋め込み外生変数生成（exog.timesfm）+ リソース計測（resources）

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-exog-timesfm \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table timesfm \
  --group-cols 'loto,ts_type' \
  --time-col ds --target-col y \
  --backend timesfm_forecast_features \
  --window-size 128 --min-points 16 --embedding-dim 256 \
  --parallel-workers 4 --enable-gpu-compute \
  --only-missing --if-exists append \
  --timesfm-codegen-yaml ./docs/lib_docs/timesfm_all_codegen.yaml
```

## Chronos埋め込み外生変数生成（exog.chronos）+ リソース計測（resources）

```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-exog-chronos \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table chronos \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y \
  --backend chronos_pipeline_auto \
  --model-id amazon/chronos-bolt-small \
  --window-size 128 --min-points 16 --embedding-dim 256 --batch-size 256 \
  --parallel-workers 4 --enable-gpu-compute \
  --only-missing --if-exists append \
  --chronos-codegen-yaml ./docs/lib_docs/chronos-forecasting_scripts_evaluation_agg-relative-score_all_codegen.yaml
```

## 統合学習データセット作成（dataset + hist + exog.*）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-unified-dataset \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --base-schema dataset --base-table loto_y_ts \
  --hist-schema dataset --hist-table loto_hist_feat \
  --exog-schema exog \
  --output-schema dataset --output-table loto_y_ts_unified \
  --output-csv-path ./artifacts/datasets/loto_y_ts_unified.csv \
  --output-parquet-path ./artifacts/datasets/loto_y_ts_unified.parquet
```

## メタテーブル駆動の網羅/再帰実行（meta.nf_automodel -> model.nf_automodel）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli db-init
PYTHONPATH=src uv run python -m loto_forecast.cli meta-automodel-run --limit 100
PYTHONPATH=src uv run python -m loto_forecast.cli meta-automodel-run --config-id 1 --stop-on-error
PYTHONPATH=src uv run python -m loto_forecast.cli meta-automodel-arg-spec --model-name AutoNHITS
```

## メタ定義作成/更新（meta.nf_automodel）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli meta-automodel-create \
  --config-name local_nf_run_01 \
  --base-schema dataset --base-table loto_y_ts_unified \
  --hist-schema dataset --hist-table loto_hist_feat \
  --exog-schema exog \
  --output-schema dataset --output-table loto_y_ts_unified \
  --unified-filter-json '{"loto":"bingo5","unique_id":"N1","ts_type":"raw"}' \
  --unified-group-cols-json '["loto","unique_id","ts_type"]' \
  --no-unified-group-validate-strict \
  --model-name AutoNHITS --h 28 \
  --auto-cls-model AutoNHITS --auto-h 28 \
  --auto-loss MAE --auto-valid-loss MAE \
  --auto-config-json '{"backend":"optuna","num_samples":10}' \
  --auto-search-alg BasicVariantGenerator \
  --auto-num-samples 10 --auto-backend optuna \
  --model-params-json '{"backend":"optuna","num_samples":20}' \
  --param-space-json '{"num_samples":[10,20],"seed":[1,2]}' \
  --recursive-depth 2 --run-predict --run-evaluate --run-explain \
  --run-save --run-load --run-analyze \
  --save-dataset --save-overwrite \
  --save-path './artifacts/saved_models/{run_id}' \
  --no-load-check-predict
```
新カラム（`auto_cls_model`, `auto_h`, `auto_config_json`）利用前に `PYTHONPATH=src uv run python -m loto_forecast.cli db-init` を実行してください。
`meta-automodel-create` は保存前に引数検証を実施し、`model_params_json` / `auto_config_json` / `param_space_json` のキー過不足・型不一致がある場合はエラーで保存を中断します。

## モデル保存/ロード/解析（NeuralForecast.save/load）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli model-save-load-analyze \
  --run-id <RUN_ID> \
  --source-path ./artifacts/<RUN_ID> \
  --save-path ./artifacts/saved_models/<RUN_ID> \
  --run-save --run-load --run-analyze \
  --save-dataset --save-overwrite \
  --no-load-check-predict
```

## PySparkでテーブル実行（JDBC + Spark SQL）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli run-table-pyspark \
  --profile local --env LOCAL \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts_unified \
  --source-sql "SELECT * FROM \"dataset\".\"loto_y_ts_unified\" WHERE y IS NOT NULL AND loto = 'bingo5' AND unique_id = 'N1' AND ts_type = 'raw'" \
  --target-schema dataset --target-table loto_y_ts_unified_spark \
  --output-if-exists replace \
  --output-parquet-path ./artifacts/datasets/loto_y_ts_unified_spark.parquet \
  --execution-backend auto --skip-row-count --no-spark-ui-enabled \
  --postgres-write-mode copy --postgres-copy-chunk-rows 50000
```
`--source-sql` を使うと PostgreSQL 側で絞り込んでから処理でき、`--transform-sql` より高速です。
`--transform-sql` が `SELECT * FROM {{source}} WHERE ...` 形式なら自動的に pushdown されます。
`--execution-backend auto` は利用可能なら `polars`/`dask` を優先し、未導入時は `pandas` に自動フォールバックします。
```bash
uv add polars "dask[dataframe]"
```

## unified dataset のグループ確認（loto,unique_id,ts_type）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli check-unified-grouping \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --schema dataset --table loto_y_ts_unified \
  --group-cols loto,unique_id,ts_type \
  --time-col ds
```

Spark起動トラブル時の確認:
```bash
echo "$SPARK_HOME"
unset SPARK_HOME
PYTHONPATH=src uv run python -m loto_forecast.cli run-table-pyspark ...省略...
```
実装側で `SPARK_HOME` 無効化再試行と `pandas` フォールバックを持つため、JSONで原因を確認できます。
`org.postgresql.Driver` が見つからない場合も、`fallback_to_pandas=true` なら自動でフォールバックされます。
`SparkUI` のポート競合を避けるには `--no-spark-ui-enabled` を指定してください。

## 実行スクリプト（scripts/*.sh）
```bash
bash scripts/run_meta_automodel_create.sh
bash scripts/run_local_nf_meta_create_and_pyspark.sh
bash scripts/run_table_pyspark.sh
bash scripts/run_fast_meta_pipeline.sh
bash scripts/run_model_save_load_analyze.sh
bash scripts/run_local_nf_full_pipeline.sh
./scripts/run_all.sh
```
`run_local_nf_meta_create_and_pyspark.sh` は `meta-automodel-create -> run-table-pyspark -> check-unified-grouping -> meta-automodel-run` を実行します。
`RUN_CHECK_GROUPING=false` でグループチェックをスキップ、`RUN_META_AUTOMODEL_RUN=false` で学習実行をスキップできます。
既定ターゲットは `TARGET_LOTO=bingo5`, `TARGET_UNIQUE_ID=N1`, `TARGET_TS_TYPE=raw` です。
stepごとに `START/DONE/FAIL`・経過秒・ログファイル（`logs/pipeline_runs`）を表示します。
`run_local_nf_full_pipeline.sh` は上記に加え `run_model_save_load_analyze.sh` を連続実行し、学習後の save/load/analyze を一括実行します。

JSON/SQL環境変数を上書きする場合の例:
```bash
MODEL_PARAMS_JSON='{"backend":"optuna","num_samples":20}' bash scripts/run_meta_automodel_create.sh
SOURCE_SQL='SELECT * FROM "dataset"."loto_y_ts_unified" WHERE y IS NOT NULL' bash scripts/run_table_pyspark.sh
RUN_ID='cfg2_d1_t1_20260219_180000' bash scripts/run_model_save_load_analyze.sh
CONFIG_NAME='local_nf_run_01' SAVE_PATH='./artifacts/saved_models/{run_id}' bash scripts/run_model_save_load_analyze.sh
```

主な環境変数:
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `CONFIG_NAME`, `MODEL_NAME`, `HORIZON`, `META_LIMIT`
- `AUTO_LOSS`, `AUTO_VALID_LOSS`, `AUTO_SEARCH_ALG`, `AUTO_NUM_SAMPLES`, `AUTO_BACKEND`
- `UNIFIED_FILTER_JSON`, `UNIFIED_GROUP_COLS_JSON`, `UNIFIED_GROUP_VALIDATE_STRICT`
- `AUTO_CLS_MODEL`, `AUTO_H`, `AUTO_CONFIG_JSON`, `AUTO_CPUS`, `AUTO_GPUS`, `MAX_TASKS`
- `RUN_SAVE`, `RUN_LOAD`, `RUN_ANALYZE`, `SAVE_DATASET`, `SAVE_OVERWRITE`, `SAVE_PATH`
- `RUN_ID`, `AUTO_RESOLVE_RUN_ID`, `CONFIG_NAME`, `CONFIG_ID`, `RUN_STATUS`, `INSAMPLE_STEP_SIZE`
- `RUN_MODEL_OPS_AFTER`, `MODEL_OPS_*`（`run_local_nf_full_pipeline.sh` 用）
- `STALE_PROCESS_POLICY`（`kill|warn|ignore`、既定`kill`。停止ジョブ由来ロック待ちの回避）
- `PROGRESS_HEARTBEAT_SECONDS`（既定10秒。無音区間の進捗ハートビート表示間隔）
- `SOURCE_SCHEMA`, `SOURCE_TABLE`, `TARGET_SCHEMA`, `TARGET_TABLE`
- `EXECUTION_BACKEND` (`auto|polars|dask|pandas|spark`), `DASK_NPARTITIONS`, `PREFER_PANDAS`
- `POSTGRES_WRITE_MODE`, `POSTGRES_COPY_CHUNK_ROWS`, `POSTGRES_LOCK_TIMEOUT_MS`

## Notebook実行
```bash
jupyter notebook notebooks/10_meta_pyspark_runner.ipynb
jupyter notebook notebooks/11_model_save_load_analyze.ipynb
jupyter notebook notebooks/12_meta_automodel_progress_runner.ipynb
jupyter notebook notebooks/13_local_full_pipeline_runner.ipynb
```
