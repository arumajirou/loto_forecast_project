# DBスキーマ定義と権限マップ

## 接続情報

```
host: 127.0.0.1
port: 5432
user: loto
dbname: loto
password: (env var DB_PASSWORD, default: z)
```

## スキーマ権限マップ

| スキーマ | 読み取り | 書き込み | DROP/TRUNCATE | 用途 |
|---------|---------|---------|--------------|------|
| `dataset` | ✅ | ❌ | ❌ | ソースデータ (変更不可) |
| `meta` | ✅ | ✅ | ❌ | 実行メタデータ |
| `model` | ✅ | ✅ | ❌ | モデル実行結果 |
| `exog` | ✅ | ✅ | ❌ | 外生変数テーブル |
| `resources` | ✅ | ✅ | ❌ | リソース監視データ |
| `catalog` | ✅ | ✅ | ❌ | codegen YAMLカタログ |
| `log` | ✅ | ✅ | ❌ | 実行ログ・UIスナップショット |

## テーブル定義

### dataset スキーマ (Read-only)

```sql
-- ソースデータ (変更不可)
dataset.loto_y_ts (unique_id TEXT, ds DATE, y FLOAT)

-- 実行メタ (model_runはmetaスキーマに移行済みだが旧テーブルも存在)
dataset.model_run (run_id UUID PK, model_name, meta JSONB, created_at)
dataset.model_metric (id, run_id FK, metric_name, metric_value, created_at)
dataset.forecast (id, run_id FK, unique_id, ds, yhat)
dataset.exog_contribution (id, run_id FK, feature_name, importance, method)
dataset.resource_sample (id, run_id FK, ts, cpu_percent, mem_percent, rss_mb)

-- グリッドサーチ
dataset.grid_search_definition (grid_id PK, model_name, horizon, param_space JSONB)
dataset.grid_search_task (id, grid_id FK, task_order, param_values JSONB, status, result, metrics, resource_summary)
dataset.execution_event_log (id, task_id, run_id, event_type, message, payload JSONB)
```

### meta スキーマ

```sql
meta.nf_automodel (
  config_id UUID PK,
  model_name TEXT,
  horizon INT,
  auto_config_json JSONB,
  param_space_json JSONB,
  param_mode_json JSONB,
  created_at TIMESTAMPTZ
)
```

### model スキーマ

```sql
model.nf_automodel (
  run_id TEXT UNIQUE,
  config_id UUID FK → meta.nf_automodel,
  status TEXT,  -- pending/running/success/failed
  params_json JSONB,
  exog_json JSONB,
  metrics_json JSONB,
  diagnostics_json JSONB,
  explain_json JSONB,
  model_path TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
)
```

### catalog スキーマ

```sql
catalog.library_catalog (id, library_name, version, description)
catalog.module_catalog (id, library_id FK, module_name, full_path)
catalog.symbol_catalog (id, module_id FK, symbol_name, full_path, parent, role, docstring, raw JSONB)
catalog.symbol_param_catalog (id, symbol_id FK, param_name, annotation, has_default, default_value, is_required)
```

### log スキーマ

```sql
log.run_history (
  history_id UUID PK, run_id TEXT, event_ts TIMESTAMPTZ,
  event_type TEXT, status TEXT, model_name TEXT, library_name TEXT,
  adapter_name TEXT, grid_id TEXT, task_id TEXT,
  horizon INT, dataset_name TEXT, log_path TEXT, message TEXT, payload JSONB
)
log.error_event (
  error_id UUID PK, run_id TEXT, event_ts TIMESTAMPTZ,
  model_name TEXT, stage TEXT, error_type TEXT, error_message TEXT, traceback TEXT, payload JSONB
)
log.ui_state_snapshot (
  state_key TEXT PK, app_name TEXT, scope TEXT, db_identity TEXT,
  state_json JSONB, state_hash TEXT, updated_at TIMESTAMPTZ
)
```

### resources スキーマ

```sql
resources.metric_def (id, name, unit, description)
resources.resource_metric (id, run_id, metric_def_id FK, ts, value)
resources.run (id, run_id UNIQUE, started_at, finished_at, status)
resources.stage_span (id, run_id FK, stage_name, started_at, finished_at, meta JSONB)
```

## よく使うクエリ

```sql
-- 最近の実行ラン
SELECT run_id, status, created_at FROM model.nf_automodel ORDER BY created_at DESC LIMIT 10;

-- グリッドサーチ状態
SELECT task_order, status, metrics FROM dataset.grid_search_task WHERE grid_id = 'xxx' ORDER BY task_order;

-- 実行エラー確認
SELECT run_id, error_type, error_message FROM log.error_event ORDER BY event_ts DESC LIMIT 10;

-- カタログ登録数
SELECT library_name, COUNT(*) FROM catalog.symbol_catalog GROUP BY library_name;
```
