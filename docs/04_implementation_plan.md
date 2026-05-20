# 実装計画書

## Phase 1（完了）

- DBメタテーブル（run/metric/forecast/exog/resource）
- AutoModel 学習/予測/評価
- ADF/Ljung-Box/Permutation

## Phase 2（完了）

- `*_all_codegen.yaml` 取り込み基盤
- `Modules/Classes/Functions/Methods/Props/External` の正規化テーブル
- 引数検証（必須漏れ/未知引数/型ヒント）

## Phase 3（完了）

- グリッド定義/タスク/実行イベントテーブル
- `grid-create` / `grid-run` / `grid-status`
- リソースサンプル・エラー・ログパスの永続化

## Phase 4（完了）

- 共通アダプタ実行層
- `retrain` コマンド
- Notebook追加（カタログ + グリッド確認）

## Phase 5（次期）

- 非同期並列ワーカー実行
- バックテスト窓の自動展開
- モデル昇格（staging/production）ワークフロー
- 複数ライブラリアダプタ（statsforecast, mlforecast 等）
