# 全体概要（拡張版）

本プロジェクトは、`dataset.loto_y_ts` を単一データソースとして以下を統合します。

1. 外生変数付き AutoModel 学習/再学習/予測/評価
2. 検定・寄与分析（ADF/Ljung-Box/Granger/Permutation）
3. グリッドサーチ実行管理（pending/running/success/failed）
4. リソース消費・実行ログ・エラー履歴の永続化
5. `*_all_codegen.yaml` を正規化して `Modules/Classes/Functions/Methods/Props/External` を DB 管理
6. ライブラリ追加時でも同一フローで実行可能な共通アダプタ構造
7. `meta.nf_automodel` 作成/更新（BaseAuto引数 `auto_*` 含む） -> 網羅/再帰実行 -> `model.nf_automodel` 保存
8. モデル保存/ロード/解析（`NeuralForecast.save/load`）の自動実行と結果回収
9. PySpark連携（PostgreSQL JDBC -> Spark SQL -> PostgreSQL/Parquet/CSV）
10. Streamlit `Operations -> Runner` による進捗バー付き実行管理
11. `scripts/*.sh` / `notebooks/*.ipynb` によるCLI・Python両面の運用導線

## 中核コンセプト

- **実行と仕様を分離**: モデル実行結果は `model_run` 系、ライブラリ仕様は `symbol_catalog` 系に分離。
- **パラメータ検証可能**: `symbol_param_catalog` を使い、必須引数漏れや未知引数を事前検知。
- **運用再現性**: run_id 単位でログ・アーティファクト・メトリクス・資源消費を追跡。
- **拡張性**: `model_registry` でアダプタを増やせば、別ライブラリも共通CLIで実行可能。
