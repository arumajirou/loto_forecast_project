# 19. NeuralForecast リソース解析 単体テスト仕様（ホワイトボックス）

## 1. 指標計算
- `duration_sec` が正しく算出される。
- `fail_rate` がゼロ割を回避して算出される。
- `throughput` が `duration_sec<=0` で `NaN` になる。

## 2. ベースライン
- グループ別 `p50/p90` が再現可能。
- ベースライン不足時にグローバル値へフォールバックする。

## 3. 異常検知
- Robust Z-score がグループ単位で計算される。
- IQR上側逸脱フラグが期待通りになる。
- `anomaly_flag` の複合条件が想定通り動作する。

## 4. 欠損耐性
- `stage_span` 欠損でも panel が落ちない。
- `run_history/error_event` 欠損でも panel が落ちない。
- `model.nf_automodel` 欠損でも run分析を継続できる。
