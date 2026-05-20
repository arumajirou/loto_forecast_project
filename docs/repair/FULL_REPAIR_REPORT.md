# FULL_REPAIR_REPORT

Generated: 2026-05-20 13:00:11 JST

## 結論

`loto_forecast_project` を安全制約優先で静的フル改修しました。
DB接続、DB書き込み、`db-init` 実適用、`grid-run`、学習、E2E/ブラウザ操作、外部送信は実行していません。

## 改修サマリ

| 区分 | 内容 |
|---|---|
| 安全ゲート | `db-init` は dry-run 既定。実適用には `LOTO_ALLOW_DB_INIT=1` と `--yes-i-understand-db-init-may-write` の両方を必須化 |
| 秘密情報 | 固定DBパスワード例、ローカルPostgreSQL URL、CLI上のパスワード直指定例を除去 |
| パス | `/mnt/e/env/ts/codex` 由来の絶対パスを相対パスまたは環境変数表現へ置換 |
| スキーマ | `dataset` への新規書き込みSQLを撤去し、`meta` / `model` / `resources` / `catalog` / `log` へ分離 |
| dashboard | 存在しない `scripts/operations_dashboard.py` 参照を `src/loto_forecast/api/streamlit/operations_dashboard.py` へ修正 |
| 不要物 | cache、artifact、node_modules、local DB、Windowsメタデータ、生成CSV/HTML/log/reportを削除 |
| 構文検査 | `compileall` は PASS |
| 実行検査 | pytest/CLI smokeはサンドボックスに `sqlalchemy` 等が無いため未完了。ローカル再検証コマンドを同梱 |

## 変更ファイル数

- 変更ファイル: 78 件
- 削除パス: 25 件
- 修復後サイズ: 69.3 MB

## 主要変更ファイル

```text
.agents/skills/log-triage/SKILL.md
.agents/skills/sandbox-check/SKILL.md
.claude/settings.example.json
.claude/settings.local.json
.claude/skills/catalog.md
.claude/skills/db-check.md
.claude/skills/db-init.md
.claude/skills/evaluate.md
.claude/skills/investigate.md
.claude/skills/python-review.md
.claude/skills/retro.md
.claude/skills/ship.md
.env.example
.gitignore
Makefile
README.md
docs/10_operations_runbook.md
docs/12_commands.md
docs/23_harness_engineering_design.md
docs/claude/commands_ref.md
docs/context/03_context_packet.md
docs/context/best_practices_registry.yaml
docs/operations_dashboard_debug_runbook.md
docs/operations_dashboard_exhaustive_test_plan.md
docs/operations_dashboard_operation_manual.md
docs/ui_audit/operations_dashboard_audit.md
docs/ui_audit/operations_dashboard_coverage_plan.md
docs/ui_audit/operations_dashboard_test_matrix.md
docs/ui_audit/panel_coverage_recovery_plan.md
docs/ui_audit/panel_renderer_decomposition_plan.md
notebooks/04_exog_build_and_resources_check.ipynb
notebooks/05_timesfm_exog_build.ipynb
notebooks/07_feature.ipynb
notebooks/08_automodel.ipynb
notebooks/10_meta_pyspark_runner.ipynb
notebooks/11_model_save_load_analyze.ipynb
notebooks/12_meta_automodel_progress_runner.ipynb
notebooks/13_local_full_pipeline_runner.ipynb
notebooks/markdown.ipynb
program.md
run_operations_dashboard.sh
scripts/analyze_nf_artifact.py
scripts/bootstrap_context_lab.sh
scripts/build_unified_dataset_fast.sh
scripts/context_lab/build_context_packet.py
scripts/context_lab/common.py
scripts/context_lab/run_all.py
scripts/context_preflight.sh
scripts/run_codex_exhaustive.sh
scripts/run_fast_meta_pipeline.sh
scripts/run_local_nf_meta_create_and_pyspark.sh
scripts/run_table_pyspark.sh
sql/00_create_schema.sql
sql/01_create_meta_tables.sql
sql/02_create_catalog_and_grid_tables.sql
sql/03_create_nf_automodel_tables.sql
src/loto_forecast/analysis/nf_artifact_analysis.py
src/loto_forecast/api/streamlit/operations_dashboard.py
src/loto_forecast/catalog/codegen_catalog.py
src/loto_forecast/cli.py
src/loto_forecast/config/settings.py
src/loto_forecast/data/spark_table_runner.py
src/loto_forecast/data/unified_dataset.py
src/loto_forecast/infra/db.py
src/loto_forecast/infra/meta_store.py
src/loto_forecast/orchestration/meta_automodel.py
src/loto_forecast/orchestration/pipeline.py
src/loto_forecast/services/task_runner.py
src/resources/chronos_exog_pipeline.py
src/resources/exog_pipeline.py
src/resources/timesfm_exog_pipeline.py
src/resources/uni2ts_exog_pipeline.py
task_prompt.md
tests/e2e/conftest.py
tests/e2e/operations_dashboard_ui_check.mjs
tests/streamlit/test_operations_dashboard_apptest.py
tests/streamlit/test_operations_dashboard_redesign.py
tools/analyze_tree_counts.py
```

## 削除した主な不要物

```text
.backup
.claude/settings.local.json
.mypy_cache
.pytest_cache
2026-04-22T07-43_export.csv
2026-04-22T17-27_export.csv
=70.0.0
artifacts
autoresearch
autotimebench_all_20251022_002756.csv
autotimebench_all_20251022_002756.csv:Zone.Identifier
data
logs
node_modules
outputs
reports
results.tsv
score_distribution_all.html
score_distribution_all.html:Zone.Identifier
tmp_windows_chrome_pipe_probe.js
tmp_windows_chrome_probe.js
uid_stats_summary_20250925_032527_feature_leaderboards_long.csv
uid_stats_summary_20250925_032527_feature_leaderboards_long.csv:Zone.Identifier
uid_stats_summary_20250925_032527_table_index_map.csv
uid_stats_summary_20250925_032527_table_index_map.csv:Zone.Identifier
```

## 残リスク

1. `pytest`、`ruff`、`mypy`、`bandit` は依存関係を入れたユーザー環境で再実行が必要です。
2. SQLスキーマ移行は静的に修正しましたが、既存DBからの移行は別途バックアップ・差分確認が必要です。
3. `operations_dashboard.py` は巨大ファイルのままです。今回は壊れた起動パスと安全導線を優先し、分割リファクタは次フェーズ扱いにしています。
4. `--password` 文字列は安全テスト・禁止ルール文脈には残しています。実行コマンド・スクリプトからは除去しています。
