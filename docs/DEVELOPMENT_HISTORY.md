# Development History

## Goal
- 時系列予測パイプラインの運用状況を、DBメタ情報・実行履歴・外生変数テーブル・コード構造まで一元的に確認できる状態にする。
- 実装者が次の改善ポイントをすぐ特定できるよう、履歴・ヒント・可視化を同じ画面で扱えるようにする。

## Milestones

### 2026-02-19
- `build-exog` に外生特徴群を段階追加
  - `pyod`, `merlion`, `pypots`, `tsfel`, `autogluon`, `stumpy`, `tsfresh`
- すべて `hist_ / stat_ / feat_` 接頭辞で統一し、`resources.*` 計測連携を維持
- 単体テストを拡張し、`tests/unit` 全体を通過
- Streamlit運用ダッシュボードを実装
  - 実行履歴、テーブル情報、エクスポート、ディレクトリ集約、コード解析可視化
  - ディレクトリ構造差分・ファイル差分（unified diff）表示/出力
  - Command Lab（CLI解析、引数説明、コマンド生成、.sh/.py生成、実行）
  - Command実行時のライブ進捗バー、経過時間、stdout/stderr tail 表示
  - `md/json/html/mmd/csv` リッチ表示強化
  - Streamlit 互換対応（`use_container_width` 廃止対応）
  - UUID列表示時の Arrow 変換エラー回避
  - テーマをStreamlitデフォルトに回帰
  - 外部ターゲット `trend` / `timesfm` の専用解析タブを追加
  - `resources.*` テーブル分析可視化タブを追加
  - Markdown資料コンパイルタブを追加
  - 機能解説（Feature Guide）タブを追加
  - `Operations -> Runner` を追加（`meta-automodel-create` / `run-table-pyspark` / Fast Sequence）
- Meta/PySpark実行系を追加
  - `meta-automodel-create` CLI を追加
  - `run-table-pyspark` CLI を追加
  - `scripts/run_meta_automodel_create.sh` を追加
  - `scripts/run_table_pyspark.sh` を追加
  - `scripts/run_fast_meta_pipeline.sh` を追加
  - `scripts/run_model_save_load_analyze.sh` を追加
  - `notebooks/10_meta_pyspark_runner.ipynb` を追加
  - `scripts/run_all.sh` を実行可能化
  - `scripts/*.sh` の JSON/SQL デフォルト引数クォート不備を修正
  - `run-table-pyspark` は Spark 実行中エラー（例: `org.postgresql.Driver` 不足）でも pandas フォールバックに移行
  - CLI JSON引数を `json` + `ast.literal_eval` で読み込み耐性を向上
  - `meta.nf_automodel` に BaseAuto引数カラム（`auto_*`）を追加
  - `meta.nf_automodel` に BaseAuto互換カラム（`auto_cls_model`, `auto_h`, `auto_config_json`）を追加
  - `meta.nf_automodel` に `unified_group_cols_json` / `unified_group_validate_strict` を追加し、`loto_y_ts_unified` の group(+ds) 検証を導入
  - `meta.nf_automodel` に `unified_filter_json` を追加し、`{"loto":"bingo5","unique_id":"N1","ts_type":"raw"}` などの系列絞り込みを導入
  - `scripts/run_local_nf_meta_create_and_pyspark.sh` を進捗強化（`START/DONE/FAIL`, 経過秒, ログ保存）
  - `run-table-pyspark` を高速化: `prefer_pandas`, `skip_row_count`, `spark_ui_enabled`, `spark_shuffle_partitions`, `spark_reader_fetchsize`
  - `run-table-pyspark` に `transform_sql` (`SELECT * FROM {{source}} WHERE ...`) の自動pushdownを追加し、JDBC全件読込を回避
  - `run-table-pyspark` に `execution_backend`（`auto|polars|dask|pandas|spark`）を追加し、`auto` で高速バックエンド自動選択
  - `run-table-pyspark` は stage timing を JSON 出力（`stage_timings_sec`）
  - `meta.nf_automodel` に save/load/analyze 制御カラムを追加
  - `model.nf_automodel` に `model_save_json/model_load_json/model_analyze_json/model_store_path` を追加
  - `model-save-load-analyze` CLI を追加
  - `Operations -> Runner` の進捗表示を強化（ステージ表示、ログ行数、最終ログ時刻、経過秒）
  - `scripts/run_local_nf_meta_create_and_pyspark.sh` を追加
  - `scripts/run_local_nf_meta_create_and_pyspark.sh` に `meta-automodel-run` ステップを追加（`config_name` から `config_id` 自動解決）
  - `scripts/run_local_nf_full_pipeline.sh` を追加（学習/予測/評価 + save/load/analyze 一括）
  - `scripts/run_local_nf_full_pipeline.sh` に stale停止ジョブのpreflight検知/掃除を追加（`STALE_PROCESS_POLICY`）
  - `scripts/run_model_save_load_analyze.sh` を拡張（`RUN_ID` 自動解決、`{run_id}` テンプレート対応）
  - `meta-automodel-run` に逐次進捗ログを追加（config/task単位の開始/成功/失敗）
  - `meta-automodel-run` に unified build の内部進捗ログを追加（exog結合/出力処理）
  - `meta-automodel-run` で `base_table` が `*_unified` / `*_spark` の場合、`hist/exog` 再結合を自動スキップ（高速化/列爆発抑制）
  - `meta-automodel-run` で PostgreSQL 1600列上限超過時の自動フォールバックを追加（Postgres永続化をスキップして学習継続）
  - `build_automodel` から `random_seed` 非対応引数を除去し、`AutoNHITS.__init__()` 型エラーを解消
  - `build_automodel` でモデル非対応の exog 引数（`hist_exog_list` など）を自動除外
  - `train_automodel` で `random_seed` を `seed` エイリアスとして受理
  - `meta-automodel-create` に引数検証を追加（キー過不足・型不一致・param_space形状）
  - `meta-automodel-arg-spec` CLI を追加（モデル別の許可引数/必須引数/型仕様の対応表を出力）
  - `meta-automodel-run` CLI は失敗runがある場合に非0終了（`--allow-failures` で従来挙動）
  - `run_local_nf_meta_create_and_pyspark.sh` の既定 `BASE_TABLE` を `loto_y_ts_unified_spark` に変更（読込量削減）
  - `Operations -> Runner` を拡張
    - `meta-automodel-run` 単体実行
    - `meta/model` ライブステータス表示
    - `run_local_nf_full_pipeline.sh` 実行UI
    - 複合bashの step進捗推定（`[n/m] START/RUNNING/DONE/FAIL`）
  - `notebooks/13_local_full_pipeline_runner.ipynb` を追加

## Current Dashboard Scope
- `resources.run / resources.stage_span / resources.resource_metric`
- `exog.*` テーブルの列情報、件数、サンプル
- `dataset.model_run / dataset.grid_search_* / dataset.execution_event_log`
- `artifacts/`, `logs/`, `docs/lib_docs/*_all_codegen.yaml`
- コード解析（Mermaidフロー/シーケンス、ネットワーク、サンバースト）
- ディレクトリ構造差分・ファイル差分（unified diff）
- コマンドビルダー/スクリプト生成/実行ランナー
- `meta-automodel-create` / `run-table-pyspark` のGUI実行と進捗監視
- `trend` / `timesfm` 外部ディレクトリ探索・可視化

## Known Constraints
- DB未起動時は DB依存タブが表示不可（ログ/コード解析は利用可）
- Mermaid描画はブラウザ側でCDN取得に依存する場合がある
- 非常に大きいディレクトリの集約は `max files` 制限内で処理する

## Next Hints
- `resources.metric_def` を用いたメトリクス辞書表示を追加すると解析の意味づけが容易
- `dataset.forecast` と実測の誤差推移可視化を追加すると運用判断が早くなる
- `Directory Compiler` の差分比較（前回コンパイルとの差分）を追加すると変更検知に有効
- Mermaid図に手動編集モードを追加すると設計レビュー用途で使いやすい

## Update Template
以下をコピーして更新:

```md
### YYYY-MM-DD
- 目的:
- 変更:
- 影響:
- 残課題:
```
