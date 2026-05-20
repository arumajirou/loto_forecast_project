# 結合テスト仕様書

## シナリオA: DB -> 学習 -> 予測 -> 評価

1. `db-init`
2. `train`
3. `predict`
4. `evaluate`

期待:
- `artifacts/<run_id>/meta.json`
- `artifacts/<run_id>/forecast.parquet`
- `artifacts/<run_id>/evaluation.json`
- `dataset.model_run/model_metric/forecast` にデータ挿入

## シナリオB: codegen カタログ

1. `catalog-import`
2. `catalog-list`
3. `catalog-validate`

期待:
- `library_catalog/module_catalog/symbol_catalog/symbol_param_catalog` 挿入
- 引数検証で必須漏れ/未知引数を検出

## シナリオC: グリッド実行

1. `grid-create`
2. `grid-run`
3. `grid-status`

期待:
- `grid_search_task` の状態遷移
- `execution_event_log` の開始/成功/失敗イベント
- `resource_sample` の記録
