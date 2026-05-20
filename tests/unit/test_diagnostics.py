import pandas as pd

from loto_forecast.analysis.diagnostics import adf_test, ljung_box


def test_ljung_box_caps_lag_for_short_series():
    residuals = pd.Series([0.1, -0.2, 0.3, -0.1])
    out = ljung_box(residuals, lags=20)
    assert out["lags_used"] == 3
    assert out["nobs"] == 4
    assert "lb_stat" in out
    assert "lb_pvalue" in out


def test_ljung_box_handles_too_short_series():
    residuals = pd.Series([0.1, -0.2])
    out = ljung_box(residuals, lags=20)
    assert out["lags_used"] == 0
    assert out["nobs"] == 2
    assert out["error"] == "insufficient_data"


def test_adf_test_handles_too_short_series():
    out = adf_test(pd.Series([1.0, 1.2, 0.9]))
    assert out["nobs"] == 3
    assert out["error"] == "insufficient_data"
