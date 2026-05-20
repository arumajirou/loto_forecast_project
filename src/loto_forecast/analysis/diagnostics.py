from __future__ import annotations

import pandas as pd
from loguru import logger
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def adf_test(series: pd.Series, autolag: str = "AIC") -> dict:
    """ADF検定（Augmented Dickey-Fuller：単位根＝非定常性の検定）。"""
    x = series.dropna().astype(float).values
    nobs = int(len(x))
    if nobs < 4:
        return {
            "adf_stat": None,
            "pvalue": None,
            "nobs": nobs,
            "crit": {},
            "error": "insufficient_data",
        }
    try:
        res = adfuller(x, autolag=autolag)
    except Exception as e:
        return {
            "adf_stat": None,
            "pvalue": None,
            "nobs": nobs,
            "crit": {},
            "error": str(e),
        }
    return {
        "adf_stat": float(res[0]),
        "pvalue": float(res[1]),
        "nobs": int(res[3]),
        "crit": {k: float(v) for k, v in res[4].items()},
    }


def ljung_box(residuals: pd.Series, lags: int = 20) -> dict:
    """Ljung–Box検定（残差の自己相関の有無）。"""
    x = residuals.dropna().astype(float)
    nobs = int(len(x))
    if nobs < 3:
        return {
            "lb_stat": None,
            "lb_pvalue": None,
            "nobs": nobs,
            "lags_used": 0,
            "error": "insufficient_data",
        }
    max_lag = int(max(1, min(int(lags), nobs - 1)))
    try:
        lb = acorr_ljungbox(x, lags=[max_lag], return_df=True)
        r = lb.iloc[0].to_dict()
        out = {k: float(v) for k, v in r.items()}
        out["nobs"] = nobs
        out["lags_used"] = max_lag
        return out
    except Exception as e:
        return {
            "lb_stat": None,
            "lb_pvalue": None,
            "nobs": nobs,
            "lags_used": max_lag,
            "error": str(e),
        }


def granger_test(df: pd.DataFrame, y_col: str, x_col: str, maxlag: int = 8) -> dict:
    """Granger因果検定（xがyの予測に有用か）。注意: 因果を保証しない。"""
    sub = df[[y_col, x_col]].dropna()
    data = sub[[y_col, x_col]]  # y first
    out = {}
    try:
        res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
        for lag, r in res.items():
            p = r[0]["ssr_ftest"][1]
            out[str(lag)] = float(p)
    except Exception as e:
        logger.warning(f"granger_test failed: {e}")
    return out
