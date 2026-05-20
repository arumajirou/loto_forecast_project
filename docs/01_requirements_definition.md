# 要件定義書（詳細版）

## 1. 目的

- `dataset.loto_y_ts` から時系列データを取得し、外生変数を活用した AutoModel を用いて高効率に学習・再学習・予測を実行する。
- `neuralforecast_all_codegen.yaml` を含む `*_all_codegen.yaml` の情報を DB に正規化し、モデル/API仕様を機械可読で管理する。
- グリッドサーチの実行状態・結果・資源・エラー・ログをメタテーブルで追跡可能にする。
- 将来ライブラリ追加時も、共通実行APIで同等機能を利用できる設計にする。

## 2. 機能要件

### 2.1 データ取得
- PostgreSQL (`127.0.0.1:5432`, DB=`loto`, schema=`dataset`, table=`loto_y_ts`) から取得。
- 必須列: `unique_id`, `ds`, `y`。
- `ds` は datetime 型に正規化。

### 2.2 特徴量生成
- カレンダー: `year`, `quarter`, `month`, `weekofyear`, `day`, `dayofyear`, `dayofweek`, `is_weekend`
- 周期表現: `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos`
- 時系列派生: `lag_*`, `roll_mean/std/min/max_*`, `diff_*`
- 外生区分推定: `futr_exog`, `hist_exog`, `stat_exog`
- 外生変数接頭辞規約:
  - `hist_`: historical（lag/rolling/diff/ewm など）
  - `stat_`: static（系列固定統計量）
  - `feat_`: feature（時刻・周期・補助特徴）

### 2.3 モデル実行
- AutoModel 学習 (`AutoNHITS` など)
- 再学習（過去 run を起点に再実行）
- 予測・評価・検定
- 外生寄与分析（Permutation）
- Granger による外生スクリーニング

### 2.4 グリッドサーチ
- `param_space` を展開しタスク化。
- タスク単位で `status`, `run_id`, `result`, `metrics`, `resource_summary`, `error_message`, `log_path` を保存。
- 実行イベントを `execution_event_log` に保存。

### 2.7 外生テーブル・リソース記録
- 生成先: `exog.<table>`（デフォルト `exog.loto_y_ts_exog`）
- 監視保存先:
  - `resources.metric_def`
  - `resources.resource_metric`
  - `resources.run`
  - `resources.stage_span`
- GPU利用可能環境では GPU スナップショットと GPU 特徴量生成を有効化。

### 2.5 codegen カタログ
- YAML から `library/module/symbol/param` を取り込み。
- `symbol_type`（function/class/method/property/external 等）を保持。
- 引数必須性・型注釈・デフォルト値を保持し、呼び出し引数検証に利用。

### 2.6 可視化
- 予測値と実測値の重ね描画。
- 外生寄与の棒グラフ。
- Notebook でテーブル確認・実行確認。

## 3. 非機能要件

- 再現性: run_id / meta.json / DBメタ情報。
- 監査性: ログファイル + execution_event_log。
- 拡張性: アダプタ追加でライブラリ横断。
- 保守性: docs/tests/sql/src を役割分離。

## 4. 制約

- ローカルPostgreSQL接続前提。
- 追加ライブラリは依存インストールが必要。
- Explainability API はライブラリ版依存のため、Permutation を標準フォールバックとする。
