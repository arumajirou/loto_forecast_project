from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.settings import settings


@dataclass
class FeatureConfig:
    lags: list[int]
    windows: list[int]
    add_cyclical: bool = True
    add_diff: bool = True


def default_feature_config() -> FeatureConfig:
    return FeatureConfig(
        lags=list(settings.default_lags),
        windows=list(settings.default_windows),
        add_cyclical=True,
        add_diff=True,
    )


def add_time_features(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    """Create future-available calendar exogenous features."""
    time_col = time_col or settings.time_col
    out = df.copy()
    ds = pd.to_datetime(out[time_col])
    out["year"] = ds.dt.year
    out["quarter"] = ds.dt.quarter
    out["month"] = ds.dt.month
    out["weekofyear"] = ds.dt.isocalendar().week.astype(int)
    out["day"] = ds.dt.day
    out["dayofyear"] = ds.dt.dayofyear
    out["dayofweek"] = ds.dt.dayofweek  # Monday=0
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert periodic calendar features into sin/cos representation."""
    out = df.copy()
    if "dayofweek" in out.columns:
        out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    if "month" in out.columns:
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    if "dayofyear" in out.columns:
        out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int],
    group_col: str | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Create lag features from target history (hist_exog candidates)."""
    group_col = group_col or settings.id_col
    target_col = target_col or settings.target_col
    out = df.copy()
    out = out.sort_values([group_col, settings.time_col])
    for lag in sorted(set(int(x) for x in lags if int(x) > 0)):
        out[f"lag_{lag}"] = out.groupby(group_col)[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    group_col: str | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    group_col = group_col or settings.id_col
    target_col = target_col or settings.target_col
    out = df.copy()
    out = out.sort_values([group_col, settings.time_col])
    base = out.groupby(group_col)[target_col].shift(1)
    for w in sorted(set(int(x) for x in windows if int(x) > 1)):
        rolled = base.rolling(w)
        out[f"roll_mean_{w}"] = rolled.mean().reset_index(level=0, drop=True)
        out[f"roll_std_{w}"] = rolled.std().reset_index(level=0, drop=True)
        out[f"roll_min_{w}"] = rolled.min().reset_index(level=0, drop=True)
        out[f"roll_max_{w}"] = rolled.max().reset_index(level=0, drop=True)
    return out


def add_diff_features(
    df: pd.DataFrame,
    periods: list[int],
    group_col: str | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    group_col = group_col or settings.id_col
    target_col = target_col or settings.target_col
    out = df.copy().sort_values([group_col, settings.time_col])
    for p in sorted(set(int(x) for x in periods if int(x) > 0)):
        out[f"diff_{p}"] = out.groupby(group_col)[target_col].diff(p)
    return out


def make_future_df(
    df: pd.DataFrame,
    h: int,
    freq: str | None = None,
    id_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Generate future frame (unique_id x future ds)."""
    freq = freq or settings.freq
    id_col = id_col or settings.id_col
    time_col = time_col or settings.time_col

    if df.empty:
        return pd.DataFrame(columns=[id_col, time_col])

    last = df.groupby(id_col)[time_col].max().reset_index()
    if last.empty:
        return pd.DataFrame(columns=[id_col, time_col])
    frames = []
    for _, row in last.iterrows():
        uid = row[id_col]
        start = pd.to_datetime(row[time_col])
        future_ds = pd.date_range(start=start, periods=h + 1, freq=freq, inclusive="right")
        frames.append(pd.DataFrame({id_col: uid, time_col: future_ds}))
    if not frames:
        return pd.DataFrame(columns=[id_col, time_col])
    return pd.concat(frames, ignore_index=True)


def infer_exog_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Infer exogenous role from naming conventions.

    Prefix-first rules:
    - feat_* -> futr_exog
    - hist_* -> hist_exog
    - stat_* -> stat_exog

    Backward compatibility:
    - future calendar/cyclical feature names -> futr_exog
    - lag/rolling/diff names -> hist_exog
    - uncategorized numeric-like covariates -> hist_exog
    """
    core = {settings.id_col, settings.time_col, settings.target_col}
    cols = [c for c in df.columns if c not in core]
    numeric_cols = [
        c
        for c in cols
        if (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])) and df[c].notna().any()
    ]

    futr = [c for c in numeric_cols if c.startswith("feat_")]
    hist = [c for c in numeric_cols if c.startswith("hist_")]
    stat = [c for c in numeric_cols if c.startswith("stat_")]
    used = set(futr + hist + stat)

    futr_like = {
        "year",
        "quarter",
        "month",
        "weekofyear",
        "day",
        "dayofyear",
        "dayofweek",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
    }

    futr_extra = [c for c in numeric_cols if c not in used and c in futr_like]
    hist_extra = [
        c
        for c in numeric_cols
        if c not in used and (c.startswith("lag_") or c.startswith("roll_") or c.startswith("diff_"))
    ]
    other = [c for c in numeric_cols if c not in used and c not in set(futr_extra + hist_extra)]

    return {
        "futr_exog": futr + futr_extra,
        "hist_exog": hist + hist_extra + other,
        "stat_exog": stat,
    }
