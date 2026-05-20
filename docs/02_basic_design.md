# 基本設計書

## 1. 論理構成

- `data/db.py`: DB接続/クエリ
- `features/engineering.py`: 特徴量生成
- `models/neuralforecast_model.py`: AutoModel構築・学習・保存
- `orchestration/pipeline.py`: 学習/再学習/予測/評価のオーケストレーション
- `analysis/explain.py`: 外生寄与・統計分析
- `catalog/codegen_catalog.py`: YAML正規化・引数検証
- `orchestration/grid_runner.py`: グリッド定義作成・実行
- `models/registry.py`: ライブラリ横断アダプタI/F
- `infra/meta_store.py`: 実行/グリッド/イベント/資源の永続化

## 2. データフロー

1. `dataset.loto_y_ts` 読み込み
2. 特徴量生成 + exog区分推定
3. AutoModel学習
4. artifact保存 (`artifacts/<run_id>`) + DBメタ記録
5. 予測/評価/寄与分析
6. グリッドの場合はタスク表に状態遷移を記録

## 3. 拡張設計（ライブラリ追加）

- `ForecastAdapter` プロトコルに従い、以下を実装するだけでCLI統合可能。
- `list_models()`
- `validate(model_name, model_params)`
- `run(...)`

## 4. エラー制御

- 学習失敗時: `model_run.status=failed`, `error_message` 保存
- グリッド失敗時: `grid_search_task.status=failed`, イベントログ保存
- 引数異常: `catalog-validate` と adapter.validate で事前検知
