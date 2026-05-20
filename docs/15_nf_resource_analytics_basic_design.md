# 15. NeuralForecast リソース解析 基本設計

## 1. 配置
- 画面: `NeuralForecast 実行・検証ラボ` 内に `リソース解析` セクションを追加。
- 実装分割:
  - 入口: `operations_dashboard.py`
  - 本体: `dashboard_nf_resource_analytics_panel.py`

## 2. 画面構成
- ① 概要
- ② run別（ランキング）
- ③ ボトルネック
- ④ エラー
- ⑤ 比較・検定
- ⑥ 提案

## 3. 結合方針
- 主キー: `run_id::text`
- 結合対象:
  - `resources.run` (runの基底)
  - `resources.stage_span` (stage内訳)
  - `resources.resource_metric` (時系列メトリクス)
  - `log.run_history` (実行イベント)
  - `log.error_event` (エラーイベント)
  - `model.nf_automodel` (model属性; 任意)

## 4. 縮退設計
- `resources.run` がない場合は機能停止。
- それ以外のテーブルは欠損しても該当タブ/分析だけスキップ。
- 欠損理由は `st.info` で明示。
