# 総合テスト設計仕様書（ブラックボックス）

## 観点

- 正常系: 最小入力で end-to-end 完走
- 異常系: DB接続不良、必須列欠損、不正モデル名、不正引数
- 監査系: ログとDBイベントが整合するか
- 拡張系: `grid` と `catalog` が同時に運用可能か

## ケース

1. 既定設定で `train -> predict -> evaluate` が完走
2. `catalog-validate` に未知引数を与えると検出される
3. `grid-run` で1件失敗しても他タスクが実行される（`--stop-on-error` 未指定）
4. `retrain` で新 run_id が作成される
5. `logs/` に run_id 付きログが出力される
