# アーキテクチャ / データフロー図

## 全体フロー

```
┌─────────────────────────────────────────────────────┐
│  PostgreSQL loto DB                                  │
│  ┌──────────────────┐   ┌────────────────────────┐  │
│  │ dataset.loto_y_ts │   │ exog.loto_y_ts_exog    │  │
│  │ (unique_id,ds,y)  │   │ (hist_*, stat_*, feat_*)│  │
│  └────────┬─────────┘   └──────────┬─────────────┘  │
└───────────┼────────────────────────┼────────────────┘
            │ read                   │ read
            ▼                        ▼
┌───────────────────────────────────────────────────────┐
│  data/unified_dataset.py                              │
│  data/dataset_loader.py                               │
│  (backends: pandas/polars/dask/spark/ray)             │
└──────────────────────┬────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────┐
│  features/engineering.py                              │
│  prepare_dataset()                                    │
│  ├── add_calendar_features()                         │
│  ├── add_cyclical_features()                         │
│  ├── add_lag_features()                              │
│  ├── add_rolling_features()                          │
│  └── add_diff_features()                             │
└──────────────────────┬────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │ train()              │ grid search
            ▼                     ▼
┌───────────────────┐  ┌──────────────────────────────┐
│ orchestration/    │  │ orchestration/                │
│ pipeline.py       │  │ grid_runner.py               │
│ train()           │  │ create_grid()                │
│ retrain()         │  │ run_grid()                   │
└────────┬──────────┘  └──────────────┬───────────────┘
         │                            │
         ▼                            ▼
┌───────────────────────────────────────────────────────┐
│  models/neuralforecast_model.py                       │
│  models/registry.py (adapter: neuralforecast_auto)   │
│  NeuralForecast AutoNHITS/AutoPatchTST/etc.          │
└────────┬──────────────────────────────────────────────┘
         │
         ├── artifacts/          (モデルファイル保存)
         │
         ├── infra/meta_store.py → meta.nf_automodel (設定)
         │                      → model.nf_automodel (結果)
         │                      → log.run_history    (イベント)
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  analysis/                                            │
│  ├── evaluation.py  → MAE/MASE/SMAPE                │
│  ├── diagnostics.py → ADF / Ljung-Box               │
│  └── explain.py     → Permutation / Granger         │
└────────┬──────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  API Layer                                            │
│  ├── api/server.py (FastAPI)                         │
│  └── api/streamlit/operations_dashboard.py (UI)      │
└───────────────────────────────────────────────────────┘
```

## CLI コマンドとモジュールのマッピング

```
cli.py
├── db-init          → infra/db.py (execute_sql_file)
├── train            → orchestration/pipeline.py (train)
├── retrain          → orchestration/pipeline.py (retrain)
├── predict          → orchestration/pipeline.py (predict)
├── evaluate         → analysis/evaluation.py
├── explain          → analysis/explain.py
├── grid-create      → orchestration/grid_runner.py (create_grid)
├── grid-run         → orchestration/grid_runner.py (run_grid)
├── grid-status      → infra/meta_store.py (list_grid_tasks)
├── build-exog       → src/resources/exog_pipeline.py
├── build-exog-timesfm → src/resources/timesfm_exog_pipeline.py
├── build-exog-chronos → src/resources/chronos_exog_pipeline.py
├── build-exog-uni2ts  → src/resources/uni2ts_exog_pipeline.py
├── catalog-import   → catalog/codegen_catalog.py
├── catalog-validate → catalog/codegen_catalog.py
├── catalog-list     → catalog/codegen_catalog.py
└── adapters         → models/registry.py (list_adapters)
```

## フェーズ実装状態

```
Phase 1 ✅  DBメタテーブル / AutoModel学習・予測・評価 / ADF・Ljung-Box診断
Phase 2 ✅  *_all_codegen.yaml 正規化 / カタログDB / 引数検証
Phase 3 ✅  グリッド定義・タスク・イベントテーブル / grid-create/run/status コマンド
Phase 4 ✅  共通アダプタ実行層 / retrain コマンド / カタログ検証強化
Phase 5 🚧  非同期並列ワーカー / バックテストウィンドウ / モデルプロモーション
             マルチライブラリアダプタ (statsforecast / mlforecast)
```

## 独立パッケージ構成

```
src/resources/  ← 独立リソース監視パッケージ (別途 uv 管理で install 可)
├── collectors/ (psutil_collector, nvml_collector, db_collector)
├── db/         (schema.py, writer.py)
└── tests/      (4テストファイル)
```
