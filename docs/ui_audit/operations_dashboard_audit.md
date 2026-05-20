# Operations Dashboard UI/UX Audit

対象:
- `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py`
- `${PROJECT_ROOT}/run_operations_dashboard.sh`

監査日時:
- 2026-03-30

実施内容:
- `streamlit.testing.v1.AppTest` による headless 描画確認
- パネル切替と主要 widget 群の存在確認
- DB未接続時の graceful degradation 確認
- 実ブラウザ起動の事前確認

観測メモ:
- 初回 AppTest で `dashboard_arg_utils` の import 失敗を再現。`operations_dashboard.py` の import path 前提が脆弱だったため修正済み
- 同ファイル内の `PROJECT_ROOT` 解決が誤っており、repo root ではなく `src/loto_forecast/api` を指していたため修正済み
- sandbox 内では Streamlit のポート bind が拒否され、実ブラウザ起動は未完了
- ローカル環境には `playwright` / `pytest-playwright` が未導入

## 問題一覧

| 問題 | 重要度 | 発生箇所 | 原因仮説 | 改善案 | 実装コスト | リスク | 検証方法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Headless 検査で import 失敗していた | 高 | `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | 同階層モジュール import を `sys.path` 依存で解決していた | `STREAMLIT_DIR` と `SRC_ROOT` を明示的に `sys.path` へ追加し、AppTest/CLI 双方で安定化 | 小 | 低 | `pytest -q tests/streamlit/test_operations_dashboard_apptest.py` |
| repo root 解決が誤っており logs/artifacts/docs 基点がずれる | 高 | `${PROJECT_ROOT}/src/loto_forecast/api/streamlit/operations_dashboard.py` | `PROJECT_ROOT = Path(__file__).resolve().parents[1]` が誤り | `STREAMLIT_DIR` / `SRC_ROOT` / `PROJECT_ROOT` を段階的に定義し直す | 小 | 中 | AppTest 実行、ローカル path 表示、artifact/log パネルの参照確認 |
| DB未接続エラーが全パネル上部で常時強く表示される | 高 | 初期描画共通部 | DB接続結果を常に `st.error` で最上段表示している | DB依存/非依存パネルで severity と文言を分ける。非依存パネルでは `warning` または dismissible info に落とす | 小 | 低 | DB停止状態で初期画面 screenshot 比較 |
| サイドバー責務が過剰で、接続情報・性能設定・Quick Launch が混在している | 高 | サイドバー | グローバル設定と運用ユーティリティを一箇所へ追加し続けた | `接続`, `表示設定`, `開発/外部起動` に再編し、Quick Launch は別パネルへ移す | 中 | 低 | 初見ユーザーにタスク到達時間を測定 |
| `NeuralForecast 実行・検証ラボ` の情報量が多く、初見導線より実装都合の構造が勝っている | 高 | NF ラボ全体 | train/retrain/predict/save/load/効果検証/DB管理が 1 画面に集約 | `準備`, `学習`, `評価`, `保存/再利用`, `詳細分析` の5群へ分割し、danger 操作は別領域へ隔離 | 中 | 中 | train 到達までのクリック数、誤操作率 |
| Train セクションのフォームが長く、入力順が自然でない | 高 | NF ラボ `学習(train)` | 実行引数ベースで UI を積み上げたため、ユーザータスク順になっていない | `目的 -> データ -> モデル -> 探索 -> リソース -> 実行` の順へ再編。advanced は expander 化 | 中 | 低 | `学習(train)` でスクロール量、入力完了時間を比較 |
| 危険操作の確認方式が統一されていない | 高 | `Run db-init`, schema 初期化, Runner 系 | 危険度ごとに確認 UX が設計されていない | danger 操作に共通確認コンポーネントを導入し、影響範囲・対象 schema を明示 | 中 | 中 | 危険操作前に必ず確認 UI が出ることを E2E で検証 |
| 実ブラウザ E2E と console/pageerror 監視が未整備 | 高 | リポジトリ全体 | backend 品質ゲート中心で UI 動的検査が未導入 | Playwright POM と trace/screenshot 付き opt-in E2E を追加 | 中 | 低 | `RUN_STREAMLIT_E2E=1 pytest -q tests/e2e/test_operations_dashboard_playwright.py` |
| query params による deep link / 初期状態復元が実質未対応 | 中 | `main()` 入口 | `st.query_params` を使う導線がない | `panel`, `nf_section`, `run_id` などの query param を受けて初期選択に反映 | 中 | 低 | `?panel=...&nf_section=...` で正しい初期表示を確認 |
| session_state key が多く、結果 payload も保持している | 中 | NF ラボ、Runner、bundle 系 | 成功結果や DataFrame/JSON を session に残し続ける設計 | key namespace を整理し、大きい payload はファイル/キャッシュへ退避、古い結果は prune | 中 | 中 | セッション key 数、メモリ使用量、rerun 時間を比較 |
| 初回描画時に重い import と GPU/NVML 検出が走り、AppTest でもノイズが多い | 中 | module import 時点 | `torch` / `cupy` / `scipy` import が eager | lazy import 化し、GPU 情報は必要時のみ取得。`MPLCONFIGDIR` も固定化 | 中 | 低 | 初回表示時間、stderr warning 数 |
| DB未接続 fallback はあるが、どの機能が使えるかの説明が散っている | 中 | 初期画面、`運用` fallback | 代替導線が guide と warning に分散 | 非DBモード専用 callout を1箇所に集約し、使えるパネルを即リンク化 | 小 | 低 | DB停止状態のタスク完遂率を比較 |
| チャート/テーブルの単位・凡例・意図が一貫していない | 中 | リソース分析、メタ深層分析、モデル分析 | 可視化コードが用途別に増え、表示ルールが統一されていない | 共通 chart helper を導入し、軸ラベル・単位・empty state を標準化 | 中 | 低 | representative chart screenshot 比較 |
| アクセシビリティ検証が未実施 | 中 | 全体 | heading 階層・role・name を機械検証していない | Playwright accessibility snapshot を追加し、heading/label 規約を整備 | 小 | 低 | a11y snapshot 差分を CI で監視 |

## 観点別所見

### 情報設計

- 画面の役割自体は広いが、運用監視と実行 UI と資料統合が一画面に並び、メンタルモデルが分裂している
- 初見導線は書かれているが、実際のデフォルト選択は `メタテーブル確認` で、主要タスクの `学習(train)` とズレている
- サイドバーが「資格情報」「性能設定」「外部起動」を同時に抱えており、視線が分散する

### 操作性

- ボタン名は一部具体的だが、`再読込` や `ヒント` のように作用範囲が分かりにくいものが残る
- `学習(train)` は入力順より実装引数順に近く、初見で上から順に埋めても成立しにくい
- `db-init` と schema 初期化は危険度の割に説明が埋もれやすい

### 表示品質

- 情報密度は高いが、主要 CTA と補助情報が同等の重みで並ぶ
- warning / error / info の量が多く、状態の優先順位が伝わりにくい
- DB未接続時のトップエラーが強すぎて、DB非依存パネルの可用性が視覚的に伝わらない

### 状態表現

- empty / warning / error は広く実装されている点は良い
- 一方で partial failure の説明は panel ごとにばらつく
- retry 導線は `再読込` と各ボタン再実行に依存しており、状態ごとの next action が十分に統一されていない

### アクセシビリティ

- Streamlit 標準 widget に依存しているため最低限の label はある
- ただし heading 階層と role/name の機械検証が未整備
- 長大フォームで keyboard 操作を前提としたグルーピングが弱い

### 性能 / 安定性

- `ui_single_panel_mode` は有効な回避策だが、裏返すと全タブ描画が重いことの兆候
- eager import と GPU/NVML 検出、ファイル走査、複数 query 系処理が初回表示コストを押し上げる
- session_state 保持量が大きく、機能追加とともに再描画不安定性が増える構造

## 改善候補一覧

### 即効改善

| 現象 | 原因仮説 | 改善案 | 検証方法 |
| --- | --- | --- | --- |
| DB未接続エラーが強すぎる | 画面全体共通で `st.error` を使用 | 非DBパネルでは `warning/info` に落とし、使用可能機能を横に出す | DB停止状態の screenshot 比較 |
| 初見で train に到達しにくい | デフォルトメニューが `メタテーブル確認` | `学習(train)` を初期選択候補に見直すか、初回だけ明示 CTA を出す | AppTest で default 選択確認 |
| `ヒント` が多すぎる | 補助情報が逐次追加された | セクション別 help callout に集約し、個別ボタンを削減 | train 画面のボタン数比較 |
| eager import warnings が多い | `torch` / `matplotlib` 周辺の初期化 | `MPLCONFIGDIR` 固定、GPU情報 lazy 化 | 初回描画時間と stderr 行数比較 |

### 中規模改善

| 現象 | 原因仮説 | 改善案 | 検証方法 |
| --- | --- | --- | --- |
| train フォームが長い | 実行引数中心の UI 構成 | ステップ分割または section cards 化 | train 完了時間比較 |
| サイドバーが肥大 | グローバル設定の集約しすぎ | 接続/表示/外部起動へ再編 | 初見ユーザーの迷いポイント観察 |
| 状態表現が panel ごとにばらつく | 各機能が独立成長 | 共通 state renderer を作る | warning/error 文言と style の統一確認 |
| deep link 不可 | query params 未実装 | panel/section/run_id を query param へ対応 | query param 付き E2E |

### 構造改善

| 現象 | 原因仮説 | 改善案 | 検証方法 |
| --- | --- | --- | --- |
| `operations_dashboard.py` が巨大で責務過多 | 機能追加を単一ファイルへ集約 | panel 単位/フォーム単位に分割し、表示ロジックと計算ロジックを分離 | モジュールサイズ、テスト容易性の改善 |
| session_state が無秩序に増える | key 管理規約が弱い | state registry / view model / DTO 導入 | key 数と rerun 不整合の減少 |
| DB query と可視化が密結合 | UI 層で query を直接持つ | repository / service / view-model 境界を切る | unit test 化率、描画時間比較 |
| 危険操作 UX が統一されない | コマンド実行 UI が個別実装 | 共通 confirmation/danger execution framework を導入 | `db-init` / truncate 系の一貫性確認 |

## 優先順位付け

1. import/path 基盤修正と AppTest 導入
2. Playwright opt-in 基盤追加
3. DB未接続時のトップメッセージ整理
4. NF ラボ train 導線整理
5. サイドバー再編
6. session_state / query params / heavy import の構造改善

## 今回の最小実装

- `operations_dashboard.py` の import path と root path 解決を修正
- AppTest ベースの headless テストを追加
- Playwright 用の opt-in E2E scaffold を追加
- テストマトリクスと監査レポートを追加

## テスト運用メモ

- UI テスト単独実行は coverage fail-under の対象外で見るため `--no-cov` を使う
- 推奨:
  - `uv run pytest --no-cov tests/streamlit -q`
  - `uv run pytest --no-cov tests/e2e -q`
- 全体回帰は `uv run pytest -q`
