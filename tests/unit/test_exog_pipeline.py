import pandas as pd

from resources.exog_pipeline import ExogBuildSpec, build_exog_dataframe


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loto": ["A"] * 6 + ["B"] * 6,
            "unique_id": ["U1"] * 6 + ["U2"] * 6,
            "ts_type": ["daily"] * 12,
            "ds": pd.date_range("2025-01-01", periods=6, freq="D").tolist()
            + pd.date_range("2025-01-01", periods=6, freq="D").tolist(),
            "y": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15],
            "proc_seconds": [0.1] * 12,
        }
    )


def test_build_exog_prefixes() -> None:
    spec = ExogBuildSpec(parallel_workers=1, enable_gpu_compute=False)
    out = build_exog_dataframe(_sample_df(), spec)

    derived = [c for c in out.columns if c not in {"loto", "unique_id", "ts_type", "ds", "y"}]
    assert derived
    assert all(c.startswith(("hist_", "stat_", "feat_")) for c in derived)


def test_static_features_constant_within_group() -> None:
    spec = ExogBuildSpec(parallel_workers=1, enable_gpu_compute=False)
    out = build_exog_dataframe(_sample_df(), spec)

    for _, g in out.groupby(["loto", "unique_id", "ts_type"]):
        assert g["stat_y_mean"].nunique(dropna=False) == 1
        assert g["stat_y_count"].nunique(dropna=False) == 1


def test_hist_lag_feature() -> None:
    spec = ExogBuildSpec(parallel_workers=1, enable_gpu_compute=False)
    out = build_exog_dataframe(_sample_df(), spec)

    g = out[(out["loto"] == "A") & (out["unique_id"] == "U1")].sort_values("ds")
    assert pd.isna(g.iloc[0]["hist_lag_1"])
    assert g.iloc[1]["hist_lag_1"] == 1
    assert g.iloc[2]["hist_lag_1"] == 2


def test_anomaly_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        pyod_detectors=("ECOD", "IForest"),
        anomaly_min_train_size=4,
    )
    out = build_exog_dataframe(_sample_df(), spec)

    expected = {
        "hist_outlier_zscore_abs",
        "hist_outlier_flag_z3",
        "hist_outlier_iqr_score",
        "hist_outlier_flag_iqr",
        "hist_outlier_robust_z_abs",
        "hist_outlier_flag_robust",
        "hist_pyod_ecod_score",
        "hist_pyod_ecod_flag",
        "hist_pyod_iforest_score",
        "hist_pyod_iforest_flag",
    }
    assert expected.issubset(set(out.columns))


def test_outlier_feature_reacts_to_spike() -> None:
    df = pd.DataFrame(
        {
            "loto": ["A"] * 8,
            "unique_id": ["U1"] * 8,
            "ts_type": ["daily"] * 8,
            "ds": pd.date_range("2025-01-01", periods=8, freq="D"),
            "y": [1, 1, 1, 1, 100, 1, 1, 1],
        }
    )
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        pyod_detectors=(),
        anomaly_min_train_size=4,
        anomaly_rolling_window=5,
    )
    out = build_exog_dataframe(df, spec).sort_values("ds").reset_index(drop=True)

    # The spike at index=4 appears in lag-based anomaly features at index=5.
    assert out.loc[5, "hist_outlier_iqr_score"] > 0.0
    assert out.loc[5, "hist_outlier_flag_iqr"] == 1.0


def test_merlion_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        enable_anomaly_features=False,
        enable_merlion_features=True,
        merlion_models=("stat_threshold", "iforest"),
        merlion_min_train_size=4,
        merlion_n_estimators=20,
        merlion_max_n_samples=64,
    )
    out = build_exog_dataframe(_sample_df(), spec)

    expected = {
        "hist_merlion_stat_threshold_score",
        "hist_merlion_stat_threshold_flag",
        "hist_merlion_iforest_score",
        "hist_merlion_iforest_flag",
    }
    assert expected.issubset(set(out.columns))


def test_pypots_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        enable_anomaly_features=False,
        enable_merlion_features=False,
        enable_pypots_features=True,
        pypots_models=("transformer", "dlinear"),
        pypots_window_size=8,
        pypots_min_train_windows=999,
    )
    out = build_exog_dataframe(_sample_df(), spec)

    expected = {
        "hist_pypots_missing_ratio",
        "hist_pypots_missing_flag",
        "hist_pypots_transformer_score",
        "hist_pypots_transformer_flag",
        "hist_pypots_dlinear_score",
        "hist_pypots_dlinear_flag",
    }
    assert expected.issubset(set(out.columns))


def test_tsfel_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        enable_anomaly_features=False,
        enable_merlion_features=False,
        enable_pypots_features=False,
        enable_tsfel_features=True,
        tsfel_domains=("statistical",),
        tsfel_max_features=4,
        tsfel_window_size=8,
        tsfel_min_train_windows=999,
    )
    out = build_exog_dataframe(_sample_df(), spec)

    assert "hist_tsfel_missing_ratio" in out.columns
    assert "hist_tsfel_missing_flag" in out.columns
    tsfel_cols = [c for c in out.columns if c.startswith("hist_tsfel_")]
    assert len(tsfel_cols) >= 6


def test_autogluon_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        enable_anomaly_features=False,
        enable_merlion_features=False,
        enable_pypots_features=False,
        enable_tsfel_features=False,
        enable_autogluon_features=True,
        autogluon_window_size=8,
        autogluon_min_train_windows=999,
        autogluon_max_features=8,
    )
    out = build_exog_dataframe(_sample_df(), spec)

    expected = {
        "hist_autogluon_missing_ratio",
        "hist_autogluon_missing_flag",
        "hist_autogluon_raw_w_last",
        "hist_autogluon_raw_w_mean",
        "hist_autogluon_raw_w_std",
    }
    assert expected.issubset(set(out.columns))


def test_stumpy_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        enable_anomaly_features=False,
        enable_merlion_features=False,
        enable_pypots_features=False,
        enable_tsfel_features=False,
        enable_autogluon_features=False,
        enable_stumpy_features=True,
        stumpy_window_size=8,
        stumpy_min_train_windows=999,
    )
    out = build_exog_dataframe(_sample_df(), spec)

    expected = {
        "hist_stumpy_missing_ratio",
        "hist_stumpy_missing_flag",
        "hist_stumpy_mp_score",
        "hist_stumpy_mp_zscore",
        "hist_stumpy_discord_flag",
    }
    assert expected.issubset(set(out.columns))


def test_tsfresh_columns_are_created() -> None:
    spec = ExogBuildSpec(
        parallel_workers=1,
        enable_gpu_compute=False,
        enable_anomaly_features=False,
        enable_merlion_features=False,
        enable_pypots_features=False,
        enable_tsfel_features=False,
        enable_autogluon_features=False,
        enable_stumpy_features=False,
        enable_tsfresh_features=True,
        tsfresh_window_size=8,
        tsfresh_min_train_windows=999,
        tsfresh_feature_set="minimal",
    )
    out = build_exog_dataframe(_sample_df(), spec)

    expected = {
        "hist_tsfresh_missing_ratio",
        "hist_tsfresh_missing_flag",
    }
    assert expected.issubset(set(out.columns))
