# Operations Dashboard Test Matrix

対象ファイル:
- `${PROJECT_ROOT}/run_operations_dashboard.sh`
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py`

## Entrypoint

| 項目 | 内容 |
| --- | --- |
| 起動スクリプト | `streamlit run ${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` |
| Streamlit entrypoint | `main()` |
| ページ設定 | `st.set_page_config(page_title="ロト予測 運用ダッシュボード", layout="wide")` |
| 主要描画モード | `ui_single_panel_mode=True` の高速単一パネル表示、または全タブ表示 |
| 接続前提 | PostgreSQL への接続は推奨だが必須ではない。DB未接続時も一部パネルは利用可能 |
| 状態管理 | `st.session_state` に UI 設定、NeuralForecast ラボの入力、実行結果、Runner 実行状態を大量保持 |
| 外部依存 | PostgreSQL、Plotly、Torch/CuPy/SciPy、ローカル artifacts/logs/docs、外部 Streamlit アプリ起動 |

## 画面領域

| 画面領域 | 主な責務 | 主な widget | 必須入力 | データ依存 | 失敗しやすい箇所 | 監視したい指標 |
| --- | --- | --- | --- | --- | --- | --- |
| サイドバー: DB接続 | DB資格情報入力、表示上限、性能設定、Quick Launch | `text_input`, `number_input`, `slider`, `toggle`, `selectbox`, `button`, `expander` | `host`, `port`, `user`, `password`, `database` | `settings`, DB接続、GPU情報、イベントログ | 初回接続失敗、資格情報露出、設定過多、再読込で全体 rerun | 初回表示時間、DB接続成否、sidebar widget 数 |
| 概要 | スキーマ/実行状態のサマリ | `subheader`, `metric`, `dataframe`, `info` | なし | DB の `dataset/exog/resources/meta/model` | DB未接続、テーブル不存在 | overview query 時間、件数取得時間 |
| 運用 | 実行履歴、Exog、メタ、Runner、モデル解析への入口 | `tabs`, `selectbox`, `button`, `plotly_chart`, `text_input` | 一部は `run_id` | DB、artifacts、forecast parquet | タブ数過多、DB未接続時の分岐、forecast 欠損 | タブ切替 rerun 時間、例外率 |
| NeuralForecast 実行・検証ラボ | `db-init`、train/retrain/predict/evaluate/save/load/CV などの統合 UI | `selectbox`, `multiselect`, `number_input`, `text_input`, `radio`, `button`, `metric`, `expander` | セクションごとに `model`, dataset 入力, `run_id`, 各種 kwargs | DB、`model.nf_automodel`、artifacts、CLI 実行、session_state | 依存関係の密度、session_state 肥大化、入力妥当性、長時間 rerun | セクション別表示時間、session_state key 数、バリデーション警告数 |
| リソース分析 | `resources.run` をベースに可視化 | `metric`, `plotly_chart`, `selectbox` | `metric key` | DB の `resources.run` | テーブル不存在、重い集計 | リソース集計時間、plotly render 時間 |
| スキーマ出力 | テーブル一覧、SQL、export | `selectbox`, `multiselect`, `button`, `text_input` | schema/table または SQL | DB のメタ情報 | 誤 SQL、SELECT 制約、重い row count | exact count 実行時間、export 成功率 |
| DB管理/ER | DB 管理パネル | パネル側実装依存 | DB接続 | DB | パネル側の別モジュール依存 | query 時間、エラー率 |
| ディレクトリ統合 | 任意ディレクトリの payload 生成 | `text_input`, `selectbox`, `button` | `directory path` | ローカルファイル | 無効パス、重いファイル走査 | 走査時間、対象ファイル数 |
| Markdown統合 | 複数 docs の bundle 化 | `text_area/text_input`, `selectbox`, `button` | roots | ローカル markdown | 対象なし、文字コード | bundle 時間、対象ファイル数 |
| コードマップ | Mermaid/解析 | `selectbox`, `button`, `plotly_chart`, `expander` | root path | ローカルコード、Plotly | 大規模コードベース解析、plotly 未導入 | 解析時間、ノード数 |
| 外部ターゲット | `trend` / `timesfm` の参照や起動 | `tabs`, `selectbox`, `number_input`, `button` | file/app selection | 外部ディレクトリ、streamlit 起動 | 対象ディレクトリ不存在、バックグラウンド起動失敗 | 走査時間、起動成功率 |
| 成果物・ログ | artifacts/log diff の可視化 | `selectbox`, `text_input`, `checkbox`, `button` | run dir / log file | artifacts, logs, docs | ファイル不存在、diff 対象過多 | ファイル走査時間、ログ解析時間 |
| 機能ガイド | 機能説明 | `markdown`, `subheader` | なし | 静的コンテンツ | 情報量過多 | 読了導線 |
| 履歴・解説 | 開発履歴/ドキュメント参照 | `subheader`, `info` | なし | docs | docs 不在 | ドキュメント欠落率 |

## 主要ユーザーフロー

| フロー | 入口 | 主要操作 | 成功条件 | 監査ポイント |
| --- | --- | --- | --- | --- |
| DB未接続での安全起動 | サイドバー | 資格情報のまま初期表示 | 例外なし、DBエラー表示、DB非依存パネル利用可 | エラー表示のノイズ、代替導線の明確さ |
| 概要確認 | `概要` | overview 表示 | テーブル概要/指標表示または graceful warning | 説明不足、空状態 |
| Train 設定 | `NeuralForecast 実行・検証ラボ -> 学習(train)` | model, backend, dataset, filters, params 入力 | 入力群が表示され、事前チェックが妥当 | フォーム長、順序、バリデーションの理解しやすさ |
| Predict / Evaluate | `NeuralForecast 実行・検証ラボ -> 予測/評価` | `run_id` 選択、dataset 条件入力、実行 | 警告/エラーが適切 | run_id 不在、dataset 条件不足 |
| Save / Load / Analyze | `保存/ロード` | save/load 各種パス指定 | 実行前チェックが可視化 | パス入力の複雑さ |
| Runner 実行 | `運用 -> Runner` | 各 expander を開きコマンド実行 | 進捗・結果保持 | 危険操作の説明、長さ、誤操作リスク |
| ディレクトリ統合 | `ディレクトリ統合` | path 入力、preview、compile | bundle 生成 | 無効 path、結果の視認性 |
| 外部ターゲット確認 | `外部ターゲット` | file preview、compile、background launch | preview/launch 情報表示 | 外部依存の存在前提、エラーメッセージ |

## session_state 重点監視

| 系統 | 代表 key | 観点 |
| --- | --- | --- |
| グローバル UI | `ui_active_panel`, `ui_single_panel_mode`, `ui_enable_query_cache` | rerun 頻度、設定数の増加 |
| NF ラボ train | `nf_lab_train_*` | key 数が最も多く、再描画・整合性崩れの中心 |
| NF ラボ実行結果 | `nf_lab_*_result` | 成功/失敗状態の表現、古い結果の残留 |
| Runner | `runner_last_*`, `runner_async_*` | 長時間実行後の状態肥大化 |
| 補助系 | `compiled_*`, `quick_v10_launch` | 大きい payload の保持によるセッション肥大化 |

## headless/AppTest で見るべきケース

| ケース | 期待結果 |
| --- | --- |
| 初期描画 | 例外なし、タイトル/ヘッダ/サイドバー描画 |
| DB未接続 | `st.error` と代替利用案内が出る |
| `運用` パネル | DB未接続 fallback tabs が表示される |
| `NeuralForecast 実行・検証ラボ` 初期状態 | ステップ進捗、初回ナビ、`Run db-init` が表示される |
| `学習(train)` 切替 | model/backend/dataset/h 入力群と validation error が表示される |
| query params 付与 | 深刻な影響なく起動する |
| 外部/ファイル系パネル | 無効入力で info/warning/error が落ちずに出る |

## browser/E2E で見るべきケース

| ケース | 期待結果 |
| --- | --- |
| アプリ起動待ち | `/` 表示まで待機できる |
| タイトル表示 | H1 と caption が視認できる |
| サイドバー操作 | `表示パネル(高速モード)` 切替が機能する |
| `NeuralForecast 実行・検証ラボ -> 学習(train)` | 主要入力が DOM 上で操作可能 |
| `運用` fallback | DB未接続でも少なくとも 4 タブに入れる |
| コンソール健全性 | `pageerror`/重大 `console.error` なし |
| スクリーンショット | 初期画面と train セクションを保存できる |
| アクセシビリティ | snapshot が取得でき、主要 heading/フォーム名が存在する |

## 実行メモ

- UI テスト単独実行では coverage fail-under の影響を避けるため `--no-cov` を使う
- 推奨コマンド:
  - `uv run pytest --no-cov tests/streamlit -q`
  - `uv run pytest --no-cov tests/e2e -q`
  - 全体品質確認は `uv run pytest -q`
