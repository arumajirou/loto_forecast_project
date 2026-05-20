import pandas as pd

from loto_forecast.features.engineering import add_time_features, infer_exog_columns, make_future_df


def test_add_time_features():
    df = pd.DataFrame({"unique_id": ["A", "A"], "ds": ["2025-01-01", "2025-01-02"], "y": [1, 2]})
    out = add_time_features(df)
    assert "dayofweek" in out.columns


def test_make_future_df():
    df = pd.DataFrame({"unique_id": ["A"], "ds": [pd.Timestamp("2025-01-01")], "y": [1]})
    futr = make_future_df(df, h=3, freq="D")
    assert len(futr) == 3


def test_make_future_df_empty_input():
    df = pd.DataFrame({"unique_id": [], "ds": [], "y": []})
    futr = make_future_df(df, h=3, freq="D")
    assert list(futr.columns) == ["unique_id", "ds"]
    assert futr.empty


def test_infer_exog_columns_prefix_first():
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            "y": [1.0, 2.0],
            "feat_calendar": [1, 2],
            "hist_lag_1": [None, 1.0],
            "stat_group_mean": [1.5, 1.5],
            "lag_7": [None, None],
            "month": [1, 1],
            "custom_covariate": [10, 11],
        }
    )
    exog = infer_exog_columns(df)
    assert "feat_calendar" in exog["futr_exog"]
    assert "month" in exog["futr_exog"]
    assert "hist_lag_1" in exog["hist_exog"]
    assert "lag_7" not in exog["hist_exog"]
    assert "custom_covariate" in exog["hist_exog"]
    assert "stat_group_mean" in exog["stat_exog"]


def test_infer_exog_columns_ignores_non_numeric_columns():
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
            "y": [1.0, 2.0],
            "feat_num": [1, 2],
            "hist_num": [0.1, 0.2],
            "stat_num": [0.3, 0.3],
            "category_col": ["x", "y"],
            "time_like_col": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
        }
    )
    exog = infer_exog_columns(df)
    assert "feat_num" in exog["futr_exog"]
    assert "hist_num" in exog["hist_exog"]
    assert "stat_num" in exog["stat_exog"]
    assert "category_col" not in exog["futr_exog"] + exog["hist_exog"] + exog["stat_exog"]
    assert "time_like_col" not in exog["futr_exog"] + exog["hist_exog"] + exog["stat_exog"]
