# 14. NeuralForecast リソース解析 要件定義

## 1. 目的
- `run_id` 単位で `schema:log` と `schema:resources` を結合し、実行の遅延/失敗の原因を説明可能にする。
- 「遅い・落ちる」を、指標・根拠・改善アクションまで一気通貫で提示する。

## 2. 対象スコープ (MVP)
- 入力:
  - `log.run_history`
  - `log.error_event`
  - `resources.run`
  - `resources.stage_span`
  - `resources.resource_metric`
  - (任意) `model.nf_automodel`
- 出力:
  - 概要KPI（成功率、p50/p90時間、throughput、失敗率、GPU/CPU/DB関連）
  - run別サマリ（期待値との差、異常スコア）
  - ボトルネック同定（stage寄与、DB比率、GPU遊休）
  - エラー分析（例外型、頻度、直前イベント）
  - 改善提案（ルール＋統計根拠）

## 3. 非機能要件
- 画面応答性:
  - クエリ件数は `row_limit` ベースで上限管理する。
  - 重い表示はサマリ優先で段階表示する。
- 安全性:
  - SQLは固定文＋バインドパラメータを利用する。
- 欠損耐性:
  - テーブル欠損時は該当分析のみスキップし、画面は継続表示する。

## 4. 受け入れ条件
- NFラボのメニューに「リソース解析」が表示される。
- DB未接続時に安全にガード表示される。
- `run_id` 結合で、KPI/ランキング/ボトルネック/エラー/提案が表示される。
