# 21. NeuralForecast リソース解析 画面設計

## 1. 入力（上部フィルタ）
- 開始日 / 終了日
- status, app_name, model_name
- command contains
- 期待値グループキー
- ベースライン窓（直近N件 or 直近X日）

## 2. 出力
- ① 概要:
  - KPIカード
  - run別サマリ
  - 散布図・分布図
- ② run別（ランキング）:
  - 遅いrun、失敗run、低効率run、異常run
- ③ ボトルネック:
  - stage寄与(Pareto)
  - DB比率
  - GPU遊休候補
  - resource_metric波形
- ④ エラー:
  - error_type/stage別件数
  - エラー直前イベント
  - run_historyタイムライン
- ⑤ 比較・検定:
  - 群比較統計
  - Welch/Mann-Whitney/Chi-square（利用可時）
- ⑥ 提案:
  - ルール判定と改善アクション
