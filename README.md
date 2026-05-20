# loto_forecast_project

`dataset.loto_y_ts` を中心に、以下を一貫して実行する時系列予測基盤です。

- 外生変数付き AutoModel 学習 / 再学習 / 予測 / 評価
- 外生寄与分析（Permutation + Granger）
- リソース監視・実行ログ記録
- `*_all_codegen.yaml`（例: `neuralforecast_all_codegen.yaml`）のメタ情報を DB 正規化
- グリッドサーチのメタ実行テーブル管理（実行/未実行/結果/エラー/資源）
- 将来ライブラリ追加に対応した共通アダプタ実行

## Quick Start

このプロジェクトのPython仮想環境は `uv` で作成・同期します。通常は `.venv` を直接 activate せず、`uv run ...` 経由で実行してください。



> Safety defaults:
> - `dataset` schema is treated as read-only source data.
> - Set DB credentials through environment variables or `.env`; do not pass passwords on the command line.
> - `db-init` runs in dry-run mode unless you explicitly use the apply command below after backup confirmation.
> - `grid-run`, model training, DB writes and E2E/browser operations can be long-running or state-changing; confirm scope before running.


注: 複数行コマンドをそのまま貼る場合は、各行末の `\` を消さないでください。

1. uv の確認またはインストール
```bash
uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l
```

2. uv 仮想環境と依存関係（静的検査用・軽量）
```bash
./scripts/setup_uv.sh
```

手動で行う場合:
```bash
rm -f uv.lock
export UV_LINK_MODE=copy
uv venv --python 3.11
uv sync --extra dev
```

dashboardのみ使う場合:
```bash
LOTO_UV_ENV_MODE=dashboard ./scripts/setup_uv.sh
```

学習・AutoModelまで使うフル環境:
```bash
LOTO_UV_ENV_MODE=full ./scripts/setup_uv.sh
# または
./scripts/setup_uv_full.sh
```

注: 静的検査では torch / neuralforecast / CUDA 依存を入れません。`uv.lock` は利用環境で再生成します。

3. 環境変数
```bash
cp -n .env.example .env
chmod 600 .env
```

4. 静的検査
```bash
./scripts/repair_static.sh
# または確認だけなら
./scripts/verify_static.sh
```

5. DB初期化の確認（dry-runのみ。SQLは実行しません）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli db-init --dry-run
```

実DBに適用する場合は、バックアップ確認後にだけ次を実行します。
```bash
export LOTO_ALLOW_DB_INIT=1
PYTHONPATH=src uv run python -m loto_forecast.cli db-init --yes-i-understand-db-init-may-write
unset LOTO_ALLOW_DB_INIT
```

6. codegen YAML 取り込み
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli catalog-import \
  --library neuralforecast \
  --yaml-path ./docs/lib_docs/neuralforecast_all_codegen.yaml
```

7. 学習/予測/評価
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli train --model AutoNHITS --h 28 --params-json '{"num_samples":10,"seed":1}'
PYTHONPATH=src uv run python -m loto_forecast.cli predict --run-id <RUN_ID> --h 28
PYTHONPATH=src uv run python -m loto_forecast.cli evaluate --run-id <RUN_ID>
PYTHONPATH=src uv run python -m loto_forecast.cli explain --run-id <RUN_ID> --method permutation
```

7. 再学習
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli retrain --base-run-id <RUN_ID>
```

8. グリッドサーチ
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli grid-create \
  --grid-id nf_grid_001 \
  --library neuralforecast \
  --adapter neuralforecast_auto \
  --model AutoNHITS \
  --h 28 \
  --param-space-json '{"num_samples":[10,20],"seed":[1,2],"backend":["optuna"]}'

LOTO_ALLOW_GRID_RUN=1 PYTHONPATH=src uv run python -m loto_forecast.cli grid-run --grid-id nf_grid_001
PYTHONPATH=src uv run python -m loto_forecast.cli grid-status --grid-id nf_grid_001
```

8. 外生変数テーブル生成（`exog`）+ リソース登録（`resources`）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-exog \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table merlion \
  --group-cols 'loto,unique_id,ts_type' \
  --enable-anomaly-features \
  --enable-merlion-features \
  --merlion-codegen-yaml ./docs/lib_docs/merlion_dashboard_selected_codegen_details.yaml \
  --merlion-models 'iforest,lof,spectral_residual,stat_threshold' \
  --enable-pypots-features \
  --pypots-codegen-yaml ./docs/lib_docs/pypots_all_codegen.yaml \
  --pypots-models 'transformer,saits' \
  --enable-tsfel-features \
  --tsfel-codegen-yaml ./docs/lib_docs/tsfel_all_codegen.yaml \
  --tsfel-domains 'statistical,temporal,spectral' \
  --enable-autogluon-features \
  --autogluon-codegen-yaml ./docs/lib_docs/autogluon__internal__all_codegen.yaml \
  --autogluon-generators 'automl_pipeline' \
  --enable-stumpy-features \
  --stumpy-codegen-yaml ./docs/lib_docs/stumpy_all_codegen.yaml \
  --stumpy-window-size 32 \
  --stumpy-min-train-windows 20 \
  --stumpy-fill-method ffill \
  --stumpy-discord-quantile 0.98 \
  --enable-tsfresh-features \
  --tsfresh-codegen-yaml ./docs/lib_docs/tsfresh_all_codegen.yaml \
  --tsfresh-feature-set minimal \
  --tsfresh-window-size 32 \
  --tsfresh-min-train-windows 20 \
  --tsfresh-fill-method ffill \
  --tsfresh-max-features 64 \
  --tsfresh-n-jobs 0 \
  --pyod-codegen-yaml ./docs/lib_docs/pyod_all_codegen.yaml \
  --pyod-detectors 'ECOD,IForest,COPOD' \
  --parallel-workers 4 --enable-gpu-compute \
  --lib-docs-dir ./docs/lib_docs
```
`resources.run` / `resources.stage_span` / `resources.resource_metric` / `resources.metric_def` に実行メトリクスを保存します。
異常検知を有効化すると、`hist_outlier_*`（Zスコア/IQR/MAD）および `hist_pyod_*`（例: `hist_pyod_ecod_score`, `hist_pyod_iforest_flag`）が追加されます。
Merlionを有効化すると、`hist_merlion_*`（例: `hist_merlion_iforest_score`, `hist_merlion_lof_flag`）が追加されます。
PyPOTSを有効化すると、`hist_pypots_*`（例: `hist_pypots_transformer_score`, `hist_pypots_saits_flag`）が追加されます。
TSFELを有効化すると、`hist_tsfel_*`（例: `hist_tsfel_mean`, `hist_tsfel_spectral_entropy`）が追加されます。
AutoGluonを有効化すると、`hist_autogluon_*`（例: `hist_autogluon_raw_w_mean`, `hist_autogluon_auto_w_std`）が追加されます。
STUMPYを有効化すると、`hist_stumpy_*`（例: `hist_stumpy_mp_score`, `hist_stumpy_discord_flag`）が追加されます。
TSFreshを有効化すると、`hist_tsfresh_*`（例: `hist_tsfresh_missing_ratio`, `hist_tsfresh_mean`）が追加されます。

9. UNI2TS埋め込み外生変数生成（`exog.uni2ts`）+ リソース登録
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-exog-uni2ts \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts \
  --target-schema exog --target-table uni2ts \
  --group-cols 'loto,unique_id,ts_type' \
  --time-col ds --target-col y \
  --context-length 128 --embedding-dim 256 --batch-size 512 \
  --parallel-workers 4 --enable-gpu-compute \
  --uni2ts-codegen-yaml ./docs/lib_docs/uni2ts_all_codegen.yaml
```
出力列は `hist_uni2ts_1..N` に加え、`embedding_dim`, `model_name`, `model_version`, `config_hash`, `created_at`, `updated_at`, `y_idx` を含みます。

10. TimesFM埋め込み外生変数生成（`exog.timesfm`）+ リソース登録
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
出力列は `hist_timesfm_1..N` に加え、`loto_y_ts_row_id`, `embedding_dim`, `model_name`, `model_version`, `config_hash`, `created_at`, `updated_at`, `y_idx` を含みます。

11. Chronos埋め込み外生変数生成（`exog.chronos`）+ リソース登録
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
出力列は `hist_chronos_1..N` に加え、`loto_y_ts_row_id`, `embedding_dim`, `model_name`, `model_version`, `config_hash`, `created_at`, `updated_at`, `y_idx` を含みます。`zero+zscore` 前処理はベクトル化され、Chronos推論はGPUバッチ化されます。

12. 統合学習データ作成（`dataset.loto_y_ts` + `dataset.loto_hist_feat` + `exog.*` 全表）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli build-unified-dataset \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --base-schema dataset --base-table loto_y_ts \
  --hist-schema dataset --hist-table loto_hist_feat \
  --exog-schema exog \
  --output-schema dataset --output-table loto_y_ts_unified \
  --output-csv-path ./artifacts/datasets/loto_y_ts_unified.csv \
  --output-parquet-path ./artifacts/datasets/loto_y_ts_unified.parquet \
  --output-spark-path ./artifacts/datasets/loto_y_ts_unified_spark
```
外生変数区分は接頭辞ベースで推定されます:
- `stat_*` -> `stat_exog`
- `hist_*` -> `hist_exog`
- `feat_*` -> `futr_exog`

13. メタテーブル駆動の網羅/再帰 AutoModel 実行
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli db-init
PYTHONPATH=src uv run python -m loto_forecast.cli meta-automodel-run --limit 100
PYTHONPATH=src uv run python -m loto_forecast.cli meta-automodel-arg-spec --model-name AutoNHITS
```
`meta.nf_automodel` の定義を読み、`param_space_json` を展開して網羅実行します。`recursive_depth` に応じて同設定を再帰ラウンド実行し、結果は `model.nf_automodel` に保存します。

14. メタ実行定義の作成/更新（`meta.nf_automodel`）
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
  --auto-config-json '{"backend":"optuna","num_samples":10}' \
  --model-params-json '{"backend":"optuna","num_samples":20}' \
  --param-space-json '{"num_samples":[10,20],"seed":[1,2]}' \
  --recursive-depth 2 --run-predict --run-evaluate --run-explain
```
`--config-json` にJSON文字列/JSONファイルを渡すことで、メタ定義をまとめて登録できます。
新カラム（`auto_cls_model`, `auto_h`, `auto_config_json`）を使う場合は `PYTHONPATH=src uv run python -m loto_forecast.cli db-init` を先に実行してください。

15. テーブルをPySparkで実行（JDBC読込 + Spark SQL変換 + 保存）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli run-table-pyspark \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --source-schema dataset --source-table loto_y_ts_unified \
  --source-sql "SELECT * FROM \"dataset\".\"loto_y_ts_unified\" WHERE y IS NOT NULL AND loto = 'bingo5' AND unique_id = 'N1' AND ts_type = 'raw'" \
  --target-schema dataset --target-table loto_y_ts_unified_spark \
  --output-if-exists replace \
  --output-parquet-path ./artifacts/datasets/loto_y_ts_unified_spark.parquet \
  --execution-backend auto --skip-row-count --no-spark-ui-enabled \
  --postgres-write-mode copy --postgres-copy-chunk-rows 50000
```
`--execution-backend auto` は `polars -> dask -> pandas` の順で利用可能な高速経路を自動選択します。
`--source-sql` を使うと DB 側で事前フィルタされるため高速です。`transform_sql` が `SELECT * FROM {{source}} WHERE ...` 形式なら自動でpushdownされます。
`polars` / `dask` 未導入環境でも `pandas` にフォールバックして実行継続します。
```bash
uv add polars "dask[dataframe]"
```

15.1 `loto_y_ts_unified` のグループ整合性チェック（`loto,unique_id,ts_type`）
```bash
PYTHONPATH=src uv run python -m loto_forecast.cli check-unified-grouping \
  --host 127.0.0.1 --port 5432 --user loto  --database loto \
  --schema dataset --table loto_y_ts_unified \
  --group-cols loto,unique_id,ts_type \
  --time-col ds
```
`duplicate_group_time_rows=0` なら、`loto,unique_id,ts_type,ds` 単位の重複がありません。

16. Streamlitダッシュボード（操作履歴/実行結果/テーブル情報）
```bash
streamlit run scripts/operations_dashboard.py
```
`make das
hboard` でも起動できます（`.venv` が無い場合は現在の環境の `streamlit` を使用）。主な確認対象:
- `resources.run` / `resources.stage_span` / `resources.resource_metric`
- `exog.*` テーブルの列情報・件数・サンプル
- `dataset.model_run` / `dataset.grid_search_*` / `dataset.execution_event_log`
- `meta.nf_automodel` / `model.nf_automodel` / `dataset.loto_y_ts_unified`
- `artifacts/` と `logs/` のローカル実行ログ
- `streamlit_all_codegen.yaml` の要約
- スキーマ内テーブル/カラム情報の一括エクスポート（`json/csv/yaml/md/html`）
- 指定ディレクトリ内 `json/csv/yaml/md/html/mmd` の集約表示・エクスポート・読み上げ
- 複数ディレクトリの Markdown 資料コンパイル表示（`Markdown Compiler`）
- `resources.run / stage_span / resource_metric` の分析可視化（`Resources Analytics`）
- CLI解析ベースの `Command Lab`（引数の意味表示、コマンド生成、コピー、`.sh/.py` 保存、実行）
- `Operations -> Runner` で `meta-automodel-create` / `run-table-pyspark` / 高速一括実行を進捗バー付きで実行
- `Operations -> Runner` はステージ表示、stdout/stderr行数、最終ログ時刻、経過秒を表示
- `Operations -> Runner` で `model-save-load-analyze` を実行し、保存/ロード/解析をJSONで確認
- `Operations -> Runner` に `meta-automodel-run` 単体実行、`meta/model` ライブステータス表示、フルローカル一括実行を追加
- `Command Lab` の実行でも進捗バー・stdout/stderr tail を表示
- 外部 Streamlit アプリ起動（例: `/mnt/e/env/ts/lib_ana/src/ui/lib_analysis/v10/streamlit_app/app.py`）
- 外部ターゲット `trend` / `timesfm` の構成表示・コンパイル・コード解析・可視化・起動
- 機能解説タブ（`Feature Guide`）
- ディレクトリ構造差分・ファイル差分（unified diff）の可視化とエクスポート
- コード解析（Mermaidフロー/シーケンス、ネットワーク、サンバースト）
- `docs/DEVELOPMENT_HISTORY.md` の履歴・ゴール・改善ヒント表示

17. bashスクリプトで実行
```bash
bash scripts/run_meta_automodel_create.sh
bash scripts/run_local_nf_meta_create_and_pyspark.sh
bash scripts/run_table_pyspark.sh
bash scripts/run_fast_meta_pipeline.sh
bash scripts/run_model_save_load_analyze.sh
bash scripts/run_local_nf_full_pipeline.sh
./scripts/run_all.sh
```
環境変数（例: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `META_LIMIT`）で上書き可能です。
`MODEL_PARAMS_JSON` / `PARAM_SPACE_JSON` / `TRANSFORM_SQL` は JSON/SQL 文字列をシングルクォートで渡してください。
`run_local_nf_meta_create_and_pyspark.sh` は既定で `meta-automodel-create` -> `run-table-pyspark` -> `check-unified-grouping` -> `meta-automodel-run` を順に実行します。
`RUN_CHECK_GROUPING=false` でグループチェックをスキップ、`RUN_META_AUTOMODEL_RUN=false` で学習実行をスキップできます。`META_RUN_CONFIG_ID` 未指定時は `CONFIG_NAME` から自動解決して `meta-automodel-run --config-id` を実行します。
`run_local_nf_meta_create_and_pyspark.sh` の既定 `BASE_TABLE` は `loto_y_ts_unified_spark`（直前 `run-table-pyspark` で作る絞り込み済みテーブル）です。`BASE_TABLE` を上書きすると全量読込になり実行時間が増えます。
`run_local_nf_meta_create_and_pyspark.sh` の既定ターゲットは `loto=bingo5`, `unique_id=N1`, `ts_type=raw` です（`TARGET_LOTO/TARGET_UNIQUE_ID/TARGET_TS_TYPE` で変更可能）。
同スクリプトは stepごとに `START/DONE/FAIL`, 経過秒, ログ出力先（`logs/pipeline_runs`）を表示します。
`run_local_nf_full_pipeline.sh` は既定で `run_local_nf_meta_create_and_pyspark.sh` 実行後に `run_model_save_load_analyze.sh` を呼び出し、学習後の save/load/analyze を自動実行します（`RUN_MODEL_OPS_AFTER=false` で無効化）。
`run_local_nf_full_pipeline.sh` は preflight で停止ジョブ（旧 `build-unified-dataset` / `run-table-pyspark` / `meta-automodel-run`）を検知し、既定 `STALE_PROCESS_POLICY=kill` で自動掃除してロック待ちを回避します。
`run_local_nf_meta_create_and_pyspark.sh` / `run_local_nf_full_pipeline.sh` は無音区間でも `RUNNING ... elapsed=... log_lines=...` のハートビート進捗を出力します（`PROGRESS_HEARTBEAT_SECONDS` で間隔調整）。
`meta-automodel-run` は `build_unified_dataset` の内部進捗（exog結合、出力処理）を逐次ログ表示します。
`meta-automodel-run` は `base_table` が `*_unified` / `*_spark` 系のとき、重複結合を避けるため `hist/exog` 再結合を自動スキップします（高速化）。
列数がPostgreSQL上限（1600列）を超える場合は、`meta-automodel-run` がPostgres永続化を自動スキップして in-memory データで学習を継続します（停止回避）。
`SparkUI` ポート競合を避けるため、スクリプト既定では `--no-spark-ui-enabled` を使用します。
`run-table-pyspark` は Spark 側で `org.postgresql.Driver` が無い場合も `pandas` フォールバックで継続します。
`meta-automodel-create` は BaseAuto 主要項目（`auto_*`）とモデル保存/ロード/解析フラグ（`run_save` など）を設定できます。`auto_cls_model` / `auto_h` / `auto_config_json` も指定可能です。
`meta-automodel-create` は保存前に引数検証を実施し、`model_params_json` / `auto_config_json` / `param_space_json` のキー過不足・型不一致を検出した場合はエラーで保存を中断します。
`unified_group_cols_json`（既定: `["loto","unique_id","ts_type"]`）と `unified_group_validate_strict` で、`loto_y_ts_unified` のグループ整合性（`group cols + ds` 重複/欠損）チェックを制御できます。
`run_model_save_load_analyze.sh` は `RUN_ID` 未指定でも `CONFIG_NAME` / `CONFIG_ID` から最新runを自動解決できます（`AUTO_RESOLVE_RUN_ID=true`）。
`meta-automodel-run` は失敗件数が1件以上あると既定で非0終了します（`--allow-failures` 指定時のみ0終了）。

18. Notebookで実行
- `notebooks/10_meta_pyspark_runner.ipynb`
  - Python APIで `meta-automodel-create` / `run-table-pyspark` 実行
  - `scripts/*.sh` 実行例を収録
- `notebooks/11_model_save_load_analyze.ipynb`
  - `NeuralForecast.save/load` と保存済みモデル解析の実行例
  - `model-save-load-analyze` CLI と bash 実行例を収録
- `notebooks/12_meta_automodel_progress_runner.ipynb`
  - Python APIで `create/run/pyspark/save-load-analyze` を段階実行し、進捗を表示
- `notebooks/13_local_full_pipeline_runner.ipynb`
  - `run_local_nf_full_pipeline.sh` をNotebookから実行
  - 最新 `model.nf_automodel` の結果確認と model-save-load-analyze 再実行例を収録

## Notebook

- `notebooks/00_db_smoke_test.ipynb`
- `notebooks/01_train_predict_automodel.ipynb`
- `notebooks/02_explainability_and_tests.ipynb`
- `notebooks/03_catalog_and_grid_validation.ipynb`
- `notebooks/04_uni2ts_exog_build.ipynb`
- `notebooks/05_timesfm_exog_build.ipynb`
- `notebooks/06_chronos_exog_build.ipynb`
- `notebooks/10_meta_pyspark_runner.ipynb`
- `notebooks/11_model_save_load_analyze.ipynb`
- `notebooks/12_meta_automodel_progress_runner.ipynb`
- `notebooks/13_local_full_pipeline_runner.ipynb`

## ドキュメント

`docs/` に要件定義、設計、テスト仕様、運用手順、DB設計、コマンド集を配置。

## 非同期実行バックエンド（Ray + FastAPI + SQLite）

UIのフリーズ回避のため、重い学習/推論/解析を別プロセスで非同期実行できます。

- API: `src/loto_forecast/api/server.py`
- タスク実行: `src/loto_forecast/services/task_runner.py`
- リソース記録: `src/loto_forecast/services/resource_logger.py`
- SQLiteメタDB: `src/loto_forecast/infra/db.py`, `src/loto_forecast/infra/orm_models.py`
- 実測vs予測・外生分析: `src/loto_forecast/analysis/forecast_analysis.py`
- デモフック: `src/loto_forecast/pipeline_hooks.py`

### 起動

```bash
export LOTO_DB_PATH=${PROJECT_ROOT}/data/registry.sqlite
bash scripts/run_async_backend_api.sh
```

### デモジョブ投入

```bash
bash scripts/submit_demo_async_task.sh
```

またはGPU指定:

```bash
NUM_GPUS=1 bash scripts/submit_demo_async_task.sh
```

### 直接APIを叩く例

```bash
curl -X POST http://127.0.0.1:8000/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{
    "kind":"train",
    "callable":"loto_forecast.pipeline_hooks:demo_train_and_predict",
    "params":{"resource_interval_s":1.0},
    "num_cpus":4,
    "num_gpus":1
  }'
```

再帰ループ投入（seedを変えながら複数タスク投入）:

```bash
curl -X POST http://127.0.0.1:8000/loops/submit \
  -H "Content-Type: application/json" \
  -d '{
    "kind":"train",
    "callable":"loto_forecast.pipeline_hooks:demo_train_and_predict",
    "params":{"dataset_id":"demo_dataset"},
    "recursive_depth":3,
    "strategy":"seed_increment",
    "seed_key":"seed",
    "seed_start":1,
    "seed_step":1,
    "num_cpus":2,
    "num_gpus":0
  }'
```

評価結果から説明契約とドリフトを取得:

```bash
curl http://127.0.0.1:8000/evaluations/<EVAL_ID>/contract
curl http://127.0.0.1:8000/evaluations/<EVAL_ID>/drift
```

`contract` には以下を格納します。
- 点予測の要約
- Conformal風の予測区間（target coverage / empirical coverage）
- 外生変数の寄与上位（SHAP/Permutation/MI/相関の優先順）
- What-ifシナリオ感度（外生変数変化に対する予測変化推定）
- 残差診断（MAE/RMSE/Ljung-Box）
# loto_forecast_project-
# loto_forecast
# loto_forecast_project
# loto_forecast_project
# loto_forecast_project
# loto_forecast_project
# loto_forecast_project
# loto_forecast_project
# loto_forecast_project


## v6 note: non-interactive uv environment rebuild

`./scripts/setup_uv.sh` now clears `.venv` by default to avoid the interactive uv prompt when switching from `static` to `dashboard` or `full` mode. Set `LOTO_UV_CLEAR_VENV=0` only when you explicitly want to reuse the existing environment.

Dashboard mode now includes `psycopg[binary]` through base dependencies because `src/resources/db/postgres_copy.py` imports the `psycopg` package at runtime.

## v10: スクリーンショット収集・特徴量ジョブ・WSL自動化

```bash
# アプリ起動
./start_loto_app.sh

# 画面スクリーンショット/trace/HAR収集
LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV=1 ./scripts/setup_uv.sh
uv run --no-sync playwright install chromium
./scripts/capture_app_screenshots.sh --url http://localhost:8505 --max-clicks 80 --max-depth 3

# datasetを読み取り、exogへ特徴量テーブルを作るdry-run
./scripts/run_dataset_feature_table_job.sh --source-schema dataset --source-table loto_y_ts_unified --target-schema exog --target-table nf_feature_table_auto --limit 5000

# DB書き込みは明示ゲート必須
export LOTO_ALLOW_FEATURE_DB_WRITE=1
./scripts/run_dataset_feature_table_job.sh --source-schema dataset --source-table loto_y_ts_unified --target-schema exog --target-table nf_feature_table_auto --limit 5000 --yes-write
unset LOTO_ALLOW_FEATURE_DB_WRITE

# cron/WSL自動化のdry-runと導入
./scripts/install_wsl_automation.sh --all
LOTO_ALLOW_AUTOMATION_INSTALL=1 ./scripts/install_wsl_automation.sh --install --all
```


## 解析資料アップロード用ZIP

解析資料・観測ログ・スクリーンショット・HAR・trace・改修レポートをアップロード用ZIPにまとめるには:

```bash
bash ./scripts/package_analysis_upload.sh --tree --note "analysis package"
```

出力先は `artifacts/upload_packages/latest_upload_package.zip` です。
