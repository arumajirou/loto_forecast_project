# 外生変数 (exog) 命名規約

## プレフィックス体系

| プレフィックス | 種別 | NF分類 | 例 |
|-------------|------|--------|-----|
| `hist_` | 履歴系 (ラグ/ローリング/差分/EWM) | `hist_exog` | `hist_lag_7`, `hist_roll_mean_14`, `hist_diff_1` |
| `stat_` | 静的統計 (系列レベル) | `stat_exog` | `stat_mean`, `stat_std`, `stat_cv` |
| `feat_` | 時間特徴量/補助特徴量 | `futr_exog` | `feat_dayofweek`, `feat_month_sin`, `feat_is_holiday` |
| (なし) | カレンダー特徴量 | `futr_exog` | `year`, `month`, `day`, `dayofweek`, `is_weekend` |

## NeuralForecast exog 区分

| NF区分 | 説明 | 推論時に必要 |
|--------|------|------------|
| `futr_exog` | 予測期間中に既知の外生変数 | はい (horizon分の値が必要) |
| `hist_exog` | 過去のみ使用する外生変数 | いいえ |
| `stat_exog` | 系列ごとの静的特徴量 | はい (1値/系列) |

## 具体例

```
# カレンダー → futr_exog
year, month, day, dayofweek, weekofyear, dayofyear
is_weekend, quarter

# 循環エンコーディング → futr_exog
dow_sin, dow_cos          # 曜日の正弦/余弦
month_sin, month_cos      # 月の正弦/余弦
doy_sin, doy_cos          # 年間日の正弦/余弦

# ラグ特徴量 → hist_exog
hist_lag_1, hist_lag_7, hist_lag_14

# ローリング統計 → hist_exog
hist_roll_mean_7, hist_roll_std_7
hist_roll_mean_14, hist_roll_max_28

# 差分 → hist_exog
hist_diff_1, hist_diff_7

# EWM → hist_exog
hist_ewm_alpha_0.3

# 静的統計 → stat_exog
stat_mean, stat_std, stat_cv, stat_skew, stat_kurt
```

## features/engineering.py との対応

```python
# prepare_dataset() 内の主要処理
add_calendar_features()   → year, month, dayofweek, is_weekend ...
add_cyclical_features()   → dow_sin/cos, month_sin/cos ...
add_lag_features()        → hist_lag_{n}
add_rolling_features()    → hist_roll_{stat}_{window}
add_diff_features()       → hist_diff_{n}
add_ewm_features()        → hist_ewm_alpha_{a}
```

## exog テーブル

生成された外生変数は `exog.loto_y_ts_exog` テーブルに保存される。

```sql
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'exog' AND table_name = 'loto_y_ts_exog'
ORDER BY ordinal_position;
```
