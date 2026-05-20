# 単体テスト仕様書（ホワイトボックス）

## 1. `features/engineering.py`

- `add_time_features` で曜日列生成
- `make_future_df` で horizon 分の future 行生成
- `lag/rolling/diff` が系列単位で生成されること

## 2. `catalog/codegen_catalog.py`

- YAML読込の構造検証
- `parse_codegen_rows` の symbol/param 正規化
- method の `parent_symbol` 推定

## 3. `orchestration/grid_runner.py`

- `expand_param_grid` が直積を正しく展開
- `max_tasks` 制限が有効

## 4. `models/registry.py`

- 不正モデル名検出
- 不正パラメータ名検出

## 5. `orchestration/pipeline.py`

- 必須列不足時の例外
- meta.json 読込失敗時の例外

## 実装済みテスト

- `tests/unit/test_features.py`
- `tests/unit/test_codegen_catalog.py`
- `tests/unit/test_grid_runner.py`
