# 16. NeuralForecast リソース解析 詳細機能設計

## 1. 派生指標
- `duration_sec = ended_at - started_at`
- `fail_rate = rows_failed / (rows_written + rows_failed)`
- `throughput = rows_written / duration_sec`
- `db_share = db_time_ms / (duration_sec * 1000)`
- `stage_share = stage_duration_ms / sum(stage_duration_ms)`

## 2. 期待値（ベースライン）
- グループキー選択: `model_name/backend/search_alg/app_name/execution_os/status`
- ベースライン窓:
  - 直近 N件
  - 直近 X日
- 出力:
  - `expected_duration_p50/p90`
  - `expected_throughput_p50`
  - `expected_fail_rate_p50`
  - `expected_gpu_util_p50`（存在時）

## 3. 異常検知
- Robust Z-score:
  - `rz = (x - median) / (1.4826 * MAD)`
- IQRルール:
  - `x > Q3 + 1.5 * IQR`
- 判定:
  - `|duration_rz| >= 3`
  - `throughput_rz <= -3`
  - `duration_ratio_vs_expected >= 1.8`
  - IQR外れ値

## 4. ボトルネック同定
- stage別総時間シェアと累積シェア（Pareto）
- stage別 `db_share` ランキング
- 長時間かつ低GPU利用の run 検出
- `resource_metric` 波形確認

## 5. 比較・検定
- 群比較:
  - Welch t-test（SciPy有効時）
  - Mann-Whitney U（SciPy有効時）
  - Cohen's d
- 成否差:
  - カイ二乗検定（SciPy有効時）

## 6. 改善提案
- ルール:
  - 高DB比率
  - 支配stage集中
  - GPU遊休
  - 高失敗率
  - 異常run多発
- 出力: `priority/pattern/evidence/suggestion`
