from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import text
from sqlalchemy.engine import Engine
from statsmodels import api as sm
from statsmodels.stats.stattools import jarque_bera

from ..config.settings import settings
from ..data.db import make_engine

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_ident(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch == "_")
    if not cleaned:
        raise ValueError(f"invalid identifier: {value}")
    return cleaned


def _to_json_like(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default
    return default


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _flatten_dict(value: dict[str, Any], prefix: str, max_depth: int = 4) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _walk(obj: Any, pfx: str, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{pfx}.{k}" if pfx else str(k)
                if isinstance(v, dict):
                    _walk(v, key, depth + 1)
                elif isinstance(v, list):
                    out[f"{key}.__len__"] = int(len(v))
                    if len(v) > 0 and all(isinstance(x, (int, float, np.number)) for x in v):
                        arr = np.asarray(v, dtype=float)
                        out[f"{key}.__mean__"] = float(np.nanmean(arr))
                        out[f"{key}.__std__"] = float(np.nanstd(arr))
                elif _is_scalar(v):
                    out[key] = v
        elif _is_scalar(obj):
            out[pfx] = obj

    _walk(value, prefix, 0)
    return out


def _normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _prepare_flattened_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if df.empty:
        return df

    for col in ["started_at", "ended_at", "created_at", "last_run_at"]:
        if col in df.columns:
            df[col] = _normalize_datetime(df[col])

    if "started_at" in df.columns and "ended_at" in df.columns:
        df["duration_sec"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()

    json_col_prefix = {
        "params_json": "params",
        "metrics_json": "metrics",
        "diagnostics_json": "diagnostics",
        "explain_json": "explain",
        "exog_json": "exog",
        "model_save_json": "save",
        "model_load_json": "load",
        "model_analyze_json": "analyze",
    }
    for col, prefix in json_col_prefix.items():
        if col not in df.columns:
            continue
        records: list[dict[str, Any]] = []
        for v in df[col]:
            obj = _to_json_like(v, default={})
            if isinstance(obj, dict):
                records.append(_flatten_dict(obj, prefix=prefix))
            else:
                records.append({})
        flat = pd.DataFrame.from_records(records, index=df.index)
        df = pd.concat([df, flat], axis=1)

    return df


def _resolve_target_metric(df: pd.DataFrame, target_metric: str | None) -> str | None:
    if df.empty:
        return None
    raw = (target_metric or "").strip()
    candidates = [raw] if raw else []
    if raw and not raw.startswith("metrics."):
        candidates.append(f"metrics.{raw}")
    candidates.extend(["metrics.mae", "metrics.rmse", "metrics.smape", "metrics.mape"])
    for c in candidates:
        if c and c in df.columns:
            return c
    metric_cols = sorted([c for c in df.columns if c.startswith("metrics.")])
    return metric_cols[0] if metric_cols else None


def _safe_float(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def _permutation_group_mean_range_pvalue(
    labels: pd.Series,
    values: pd.Series,
    n_iter: int = 500,
    seed: int = 42,
) -> float | None:
    sub = pd.DataFrame({"label": labels, "value": values}).dropna()
    if sub.empty or sub["label"].nunique() < 2:
        return None
    grp = sub.groupby("label")["value"].mean()
    observed = float(grp.max() - grp.min())
    if observed <= 0:
        return 1.0
    y = sub["value"].to_numpy(dtype=float)
    x = sub["label"].to_numpy()
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(max(50, n_iter))):
        perm = rng.permutation(y)
        tmp = pd.DataFrame({"label": x, "value": perm}).groupby("label")["value"].mean()
        score = float(tmp.max() - tmp.min())
        if score >= observed:
            ge += 1
    return float((ge + 1) / (int(max(50, n_iter)) + 1))


def _summarize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    rows: list[dict[str, Any]] = []
    for col in metric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "metric": col,
                "count": int(s.shape[0]),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("metric").reset_index(drop=True)
    return out


def _parameter_impact(df: pd.DataFrame, target_metric: str, alpha: float) -> pd.DataFrame:
    cols = [
        "parameter",
        "groups",
        "sample_size",
        "best_value",
        "best_mean",
        "worst_value",
        "worst_mean",
        "effect_abs",
        "permutation_pvalue",
        "is_significant",
        "pearson_corr_numeric",
    ]
    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(df[target_metric], errors="coerce")
    for col in [c for c in df.columns if c.startswith("params.")]:
        sub = pd.DataFrame({"x": df[col], "y": target}).dropna()
        if sub.empty or sub["x"].nunique() < 2:
            continue
        if sub["x"].nunique() > 20:
            continue
        grouped = sub.groupby("x")["y"].agg(["count", "mean", "median"]).sort_values("mean")
        if grouped.shape[0] < 2:
            continue
        best_idx = grouped["mean"].idxmin()
        worst_idx = grouped["mean"].idxmax()
        pval = _permutation_group_mean_range_pvalue(sub["x"], sub["y"], n_iter=600)
        corr = None
        if pd.api.types.is_numeric_dtype(sub["x"]):
            corr = _safe_float(sub["x"].corr(sub["y"]))
        rows.append(
            {
                "parameter": col,
                "groups": int(grouped.shape[0]),
                "sample_size": int(sub.shape[0]),
                "best_value": str(best_idx),
                "best_mean": float(grouped.loc[best_idx, "mean"]),
                "worst_value": str(worst_idx),
                "worst_mean": float(grouped.loc[worst_idx, "mean"]),
                "effect_abs": float(abs(grouped.loc[worst_idx, "mean"] - grouped.loc[best_idx, "mean"])),
                "permutation_pvalue": pval,
                "is_significant": bool(pval is not None and pval < alpha),
                "pearson_corr_numeric": corr,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = out.sort_values(
        by=["is_significant", "effect_abs"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return out


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return None
        low = raw.lower()
        if low in {"true", "false"}:
            return low == "true"
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except Exception:
            return raw
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def _values_match(expected: Any, actual: Any, atol: float = 1e-9, rtol: float = 1e-6) -> bool:
    e = _normalize_scalar(expected)
    a = _normalize_scalar(actual)
    if e is None or a is None:
        return False
    if isinstance(e, bool) or isinstance(a, bool):
        return bool(e) == bool(a)
    if isinstance(e, (int, float)) and isinstance(a, (int, float)):
        ev = float(e)
        av = float(a)
        return abs(ev - av) <= max(atol, rtol * max(abs(ev), abs(av), 1.0))
    return str(e) == str(a)


def _first_non_null(row: pd.Series, cols: Sequence[str]) -> tuple[Any, str | None]:
    for c in cols:
        if c not in row.index:
            continue
        v = row.get(c)
        vn = _normalize_scalar(v)
        if vn is not None:
            return vn, c
    return None, None


def _build_parameter_reflection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    specs = [
        {
            "parameter": "horizon",
            "expected_cols": ["cfg_horizon", "meta_horizon", "horizon"],
            "actual_cols": ["params.h", "params.horizon", "params.auto_h"],
        },
        {
            "parameter": "model_name",
            "expected_cols": ["cfg_model_name", "meta_model_name", "model_name"],
            "actual_cols": ["params.model_name", "params.auto_cls_model", "params.alias"],
        },
        {
            "parameter": "auto_num_samples",
            "expected_cols": ["cfg_auto_num_samples", "meta_auto_num_samples"],
            "actual_cols": ["params.num_samples", "params.auto_num_samples"],
        },
        {
            "parameter": "auto_backend",
            "expected_cols": ["cfg_auto_backend", "meta_auto_backend"],
            "actual_cols": ["params.backend", "params.auto_backend"],
        },
        {
            "parameter": "auto_loss",
            "expected_cols": ["cfg_auto_loss", "meta_auto_loss"],
            "actual_cols": ["params.loss_name", "params.auto_loss"],
        },
        {
            "parameter": "auto_valid_loss",
            "expected_cols": ["cfg_auto_valid_loss", "meta_auto_valid_loss"],
            "actual_cols": ["params.valid_loss_name", "params.auto_valid_loss"],
        },
        {
            "parameter": "auto_search_alg",
            "expected_cols": ["cfg_auto_search_alg", "meta_auto_search_alg"],
            "actual_cols": ["params.search_alg_name", "params.auto_search_alg"],
        },
        {
            "parameter": "auto_cpus",
            "expected_cols": ["cfg_auto_cpus", "meta_auto_cpus"],
            "actual_cols": ["params.cpus", "params.auto_cpus"],
        },
        {
            "parameter": "auto_gpus",
            "expected_cols": ["cfg_auto_gpus", "meta_auto_gpus"],
            "actual_cols": ["params.gpus", "params.auto_gpus"],
        },
        {
            "parameter": "auto_refit_with_val",
            "expected_cols": ["cfg_auto_refit_with_val", "meta_auto_refit_with_val"],
            "actual_cols": ["params.refit_with_val", "params.auto_refit_with_val"],
        },
        {
            "parameter": "auto_verbose",
            "expected_cols": ["cfg_auto_verbose", "meta_auto_verbose"],
            "actual_cols": ["params.verbose", "params.auto_verbose"],
        },
    ]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for spec in specs:
        compared_rows = 0
        matched_rows = 0
        mismatched_rows = 0
        missing_actual_rows = 0
        first_expected = None
        first_actual = None
        for _, row in df.iterrows():
            expected, expected_col = _first_non_null(row, spec["expected_cols"])
            if expected is None:
                continue
            actual, actual_col = _first_non_null(row, spec["actual_cols"])
            compared_rows += 1
            if first_expected is None:
                first_expected = expected
            if first_actual is None and actual is not None:
                first_actual = actual
            if actual is None:
                missing_actual_rows += 1
                detail_rows.append(
                    {
                        "parameter": spec["parameter"],
                        "run_id": str(row.get("run_id") or ""),
                        "config_id": row.get("config_id"),
                        "expected_col": expected_col,
                        "expected": expected,
                        "actual_col": None,
                        "actual": None,
                        "status": "missing_actual",
                    }
                )
                continue
            if _values_match(expected, actual):
                matched_rows += 1
                continue
            mismatched_rows += 1
            detail_rows.append(
                {
                    "parameter": spec["parameter"],
                    "run_id": str(row.get("run_id") or ""),
                    "config_id": row.get("config_id"),
                    "expected_col": expected_col,
                    "expected": expected,
                    "actual_col": actual_col,
                    "actual": actual,
                    "status": "mismatch",
                }
            )
        if compared_rows <= 0:
            continue
        summary_rows.append(
            {
                "parameter": spec["parameter"],
                "compared_rows": int(compared_rows),
                "matched_rows": int(matched_rows),
                "mismatched_rows": int(mismatched_rows),
                "missing_actual_rows": int(missing_actual_rows),
                "match_rate": float(matched_rows / compared_rows),
                "mismatch_rate": float((mismatched_rows + missing_actual_rows) / compared_rows),
                "sample_expected": first_expected,
                "sample_actual": first_actual,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["mismatch_rate", "compared_rows"],
            ascending=[False, False],
        ).reset_index(drop=True)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(by=["parameter", "status", "run_id"]).reset_index(drop=True)
    return summary_df, detail_df


def _build_feature_contribution(
    df: pd.DataFrame,
    target_metric: str,
    max_features: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty or target_metric not in df.columns:
        return pd.DataFrame()

    base_cols = ["horizon", "dataset_rows", "feature_cols", "duration_sec", "model_name", "status", "config_id"]
    prefixed_cols = [
        c
        for c in df.columns
        if c.startswith("params.")
        or c.startswith("exog.")
        or c.startswith("diagnostics.")
        or c.startswith("analyze.")
        or c.startswith("explain.")
        or c.startswith("save.")
        or c.startswith("load.")
    ]
    candidate_cols = [c for c in [*base_cols, *prefixed_cols] if c in df.columns and c != target_metric]
    if len(candidate_cols) < 2:
        return pd.DataFrame()

    work = df[candidate_cols + [target_metric]].copy()
    y = pd.to_numeric(work[target_metric], errors="coerce")
    X = work[candidate_cols].copy()

    useful_cols: list[str] = []
    for c in X.columns:
        non_null = int(X[c].notna().sum())
        nunique = int(X[c].nunique(dropna=True))
        if non_null >= 10 and nunique >= 2:
            useful_cols.append(c)
    if len(useful_cols) < 2:
        return pd.DataFrame()
    X = X[useful_cols]

    X = pd.get_dummies(X, dummy_na=True, drop_first=False)
    if X.empty:
        return pd.DataFrame()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    n = int(y.shape[0])
    if n < 12 or X.shape[1] < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(max(4, min(int(n * 0.3), n - 6)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if len(train_idx) < 6 or len(test_idx) < 4:
        return pd.DataFrame()

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_r2 = float(r2_score(y_test, pred))
    test_mae = float(mean_absolute_error(y_test, pred))

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=24,
        random_state=random_state,
        n_jobs=-1,
    )
    out = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "importance_mean": perm.importances_mean.astype(float),
            "importance_std": perm.importances_std.astype(float),
        }
    )
    out = out.sort_values("importance_mean", ascending=False).head(max(1, int(max_features))).reset_index(drop=True)
    den = float(out["importance_mean"].clip(lower=0).sum())
    if den > 0:
        out["contribution_rate"] = out["importance_mean"].clip(lower=0) / den
    else:
        out["contribution_rate"] = 0.0
    out["model_r2"] = test_r2
    out["model_mae"] = test_mae
    out["n_train"] = int(len(train_idx))
    out["n_test"] = int(len(test_idx))
    return out


def _build_causal_hints(
    df: pd.DataFrame,
    target_metric: str,
    candidate_cols: list[str],
    max_rows: int = 10,
) -> pd.DataFrame:
    if df.empty or target_metric not in df.columns:
        return pd.DataFrame()

    y_raw = pd.to_numeric(df[target_metric], errors="coerce")
    controls_num = [c for c in ["horizon", "dataset_rows", "feature_cols", "duration_sec"] if c in df.columns]
    controls_cat = [c for c in ["model_name", "status"] if c in df.columns]

    rows: list[dict[str, Any]] = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        t_raw = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({"_y": y_raw, "_treat": t_raw})
        for c in controls_num + controls_cat:
            tmp[c] = df[c]
        tmp = tmp.dropna(subset=["_y", "_treat"])
        if tmp.shape[0] < 30 or tmp["_treat"].nunique() < 4:
            continue

        X_num = pd.DataFrame(index=tmp.index)
        for c in controls_num:
            s = pd.to_numeric(tmp[c], errors="coerce")
            X_num[c] = s.fillna(s.median())
        X_cat = pd.DataFrame(index=tmp.index)
        if controls_cat:
            X_cat = pd.get_dummies(tmp[controls_cat].astype(str), drop_first=True, dtype=float)
        X = pd.concat([tmp[["_treat"]].astype(float), X_num, X_cat], axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        y = tmp["_y"].astype(float)
        y_std = float(y.std(ddof=1))
        if not np.isfinite(y_std) or y_std <= 0:
            continue
        y = (y - float(y.mean())) / y_std
        t = X["_treat"].astype(float)
        t_std = float(t.std(ddof=1))
        if not np.isfinite(t_std) or t_std <= 0:
            continue
        X["_treat"] = (t - float(t.mean())) / t_std

        if X.shape[0] < (X.shape[1] + 12):
            continue
        Xd = sm.add_constant(X, has_constant="add")
        try:
            fit = sm.OLS(y.to_numpy(dtype=float), Xd.to_numpy(dtype=float)).fit(cov_type="HC3")
            param_names = ["const"] + X.columns.tolist()
            idx = param_names.index("_treat")
            coef = float(fit.params[idx])
            pvalue = float(fit.pvalues[idx])
            ci = fit.conf_int(alpha=0.05)
            ci_low = float(ci[idx, 0])
            ci_high = float(ci[idx, 1])
            rows.append(
                {
                    "treatment": str(col),
                    "std_coef": coef,
                    "pvalue": pvalue,
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                    "n": int(X.shape[0]),
                    "controls": int(max(0, X.shape[1] - 1)),
                    "significant_0.05": bool(pvalue < 0.05),
                    "direction": "positive" if coef >= 0 else "negative",
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_std_coef"] = out["std_coef"].abs()
    out = (
        out.sort_values(by=["significant_0.05", "abs_std_coef"], ascending=[False, False])
        .head(max_rows)
        .reset_index(drop=True)
    )
    return out


def _recursive_drilldown(
    df: pd.DataFrame,
    target_metric: str,
    ranked_params: list[str],
    depth: int,
    max_depth: int,
    min_group_size: int,
) -> dict[str, Any]:
    y = pd.to_numeric(df[target_metric], errors="coerce").dropna()
    node: dict[str, Any] = {
        "depth": int(depth),
        "rows": int(df.shape[0]),
        "target_mean": _safe_float(y.mean()) if not y.empty else None,
        "target_median": _safe_float(y.median()) if not y.empty else None,
        "target_std": _safe_float(y.std(ddof=1)) if y.shape[0] > 1 else 0.0,
        "children": [],
    }
    if depth >= max_depth:
        return node
    if df.shape[0] < (min_group_size * 2):
        return node

    split_param = None
    grouped_cache: dict[str, Any] = {}
    for p in ranked_params:
        if p not in df.columns:
            continue
        grp = (
            df[[p, target_metric]]
            .dropna()
            .groupby(p)[target_metric]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
        )
        grp = grp[grp["count"] >= min_group_size]
        if grp.shape[0] < 2:
            continue
        split_param = p
        grouped_cache[p] = grp
        break

    if split_param is None:
        return node

    node["split_parameter"] = split_param
    grp = grouped_cache[split_param]
    child_values = list(grp.index[:8])
    for v in child_values:
        child_df = df[df[split_param] == v]
        child = _recursive_drilldown(
            df=child_df,
            target_metric=target_metric,
            ranked_params=ranked_params,
            depth=depth + 1,
            max_depth=max_depth,
            min_group_size=min_group_size,
        )
        child["condition"] = {split_param: str(v)}
        node["children"].append(child)
    return node


def _build_stat_tests(df: pd.DataFrame, target_metric: str, alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {"alpha": float(alpha)}
    y = pd.to_numeric(df[target_metric], errors="coerce").dropna()
    if y.shape[0] >= 8:
        jb_stat, jb_pvalue, skew, kurt = jarque_bera(y.to_numpy(dtype=float))
        out["normality_jarque_bera"] = {
            "n": int(y.shape[0]),
            "jb_stat": float(jb_stat),
            "pvalue": float(jb_pvalue),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "reject_normality": bool(float(jb_pvalue) < alpha),
        }
    else:
        out["normality_jarque_bera"] = {"n": int(y.shape[0]), "error": "insufficient_data"}

    if "status" in df.columns:
        success = pd.to_numeric(df.loc[df["status"] == "success", target_metric], errors="coerce").dropna()
        failed = pd.to_numeric(df.loc[df["status"] == "failed", target_metric], errors="coerce").dropna()
        pval = None
        diff = None
        if success.shape[0] >= 2 and failed.shape[0] >= 2:
            labels = pd.Series(["success"] * success.shape[0] + ["failed"] * failed.shape[0])
            values = pd.Series(np.concatenate([success.to_numpy(dtype=float), failed.to_numpy(dtype=float)]))
            pval = _permutation_group_mean_range_pvalue(labels, values, n_iter=800)
            diff = float(success.mean() - failed.mean())
        out["status_mean_diff_test"] = {
            "success_n": int(success.shape[0]),
            "failed_n": int(failed.shape[0]),
            "mean_diff_success_minus_failed": diff,
            "permutation_pvalue": pval,
            "is_significant": bool(pval is not None and pval < alpha),
        }
        if success.shape[0] >= 3 and failed.shape[0] >= 3:
            try:
                u_stat, u_p = spstats.mannwhitneyu(
                    success.to_numpy(dtype=float),
                    failed.to_numpy(dtype=float),
                    alternative="two-sided",
                )
                out["status_mannwhitney_u"] = {
                    "u_stat": float(u_stat),
                    "pvalue": float(u_p),
                    "is_significant": bool(float(u_p) < alpha),
                }
            except Exception as e:
                out["status_mannwhitney_u"] = {"error": str(e)}

    if "status" in df.columns:
        groups = []
        for _, g in df.groupby("status"):
            s = pd.to_numeric(g[target_metric], errors="coerce").dropna()
            if s.shape[0] >= 3:
                groups.append(s.to_numpy(dtype=float))
        if len(groups) >= 2:
            try:
                lev_stat, lev_p = spstats.levene(*groups, center="median")
                out["status_levene_variance_test"] = {
                    "stat": float(lev_stat),
                    "pvalue": float(lev_p),
                    "is_significant": bool(float(lev_p) < alpha),
                }
            except Exception as e:
                out["status_levene_variance_test"] = {"error": str(e)}

    if "model_name" in df.columns:
        groups = []
        labels = []
        for k, g in df.groupby("model_name"):
            s = pd.to_numeric(g[target_metric], errors="coerce").dropna()
            if s.shape[0] >= 3:
                groups.append(s.to_numpy(dtype=float))
                labels.append(str(k))
        if len(groups) >= 2:
            try:
                kw_stat, kw_p = spstats.kruskal(*groups)
                out["model_name_kruskal"] = {
                    "groups": labels[:20],
                    "stat": float(kw_stat),
                    "pvalue": float(kw_p),
                    "is_significant": bool(float(kw_p) < alpha),
                }
            except Exception as e:
                out["model_name_kruskal"] = {"error": str(e)}

    for diag_col in ["diagnostics.lb_pvalue", "diagnostics.adf_pvalue"]:
        if diag_col not in df.columns:
            continue
        p = pd.to_numeric(df[diag_col], errors="coerce").dropna()
        if p.empty:
            continue
        key = f"{diag_col}_rejection"
        out[key] = {
            "n": int(p.shape[0]),
            "alpha": float(alpha),
            "rejection_rate": float((p < alpha).mean()),
            "median_pvalue": float(p.median()),
        }
    return out


def _save_plot_status_counts(df: pd.DataFrame, out_path: Path) -> None:
    if "status" not in df.columns:
        return
    vc = df["status"].astype(str).value_counts()
    if vc.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(vc.index.tolist(), vc.values.tolist())
    ax.set_title("Run Status Counts")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_plot_target_hist(df: pd.DataFrame, target_metric: str, out_path: Path) -> None:
    y = pd.to_numeric(df[target_metric], errors="coerce").dropna()
    if y.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y, bins=min(30, max(8, int(np.sqrt(y.shape[0])))), alpha=0.85)
    ax.set_title(f"Distribution: {target_metric}")
    ax.set_xlabel(target_metric)
    ax.set_ylabel("frequency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_plot_target_over_time(df: pd.DataFrame, target_metric: str, out_path: Path) -> None:
    if "created_at" not in df.columns:
        return
    tmp = df[["created_at", target_metric]].copy()
    tmp[target_metric] = pd.to_numeric(tmp[target_metric], errors="coerce")
    tmp = tmp.dropna().sort_values("created_at")
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tmp["created_at"], tmp[target_metric], marker="o", linewidth=1.2, markersize=3)
    ax.set_title(f"{target_metric} Over Time")
    ax.set_xlabel("created_at")
    ax.set_ylabel(target_metric)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_plot_top_param_effects(impact_df: pd.DataFrame, out_path: Path, top_k: int = 12) -> None:
    if impact_df.empty:
        return
    top = impact_df.head(max(1, int(top_k))).copy()
    fig, ax = plt.subplots(figsize=(10, max(4, top.shape[0] * 0.35)))
    ax.barh(top["parameter"].astype(str), top["effect_abs"].astype(float))
    ax.invert_yaxis()
    ax.set_title("Top Parameter Effects (absolute)")
    ax.set_xlabel("effect_abs")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _build_key_insights(
    df: pd.DataFrame,
    target_metric: str,
    impact_df: pd.DataFrame,
    reflection_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    causal_hints_df: pd.DataFrame,
    higher_is_better: bool,
) -> list[str]:
    insights: list[str] = []
    if df.empty:
        return insights

    if "status" in df.columns:
        status_counts = df["status"].astype(str).value_counts()
        total = int(status_counts.sum())
        failed = int(status_counts.get("failed", 0))
        insights.append(f"失敗率: {failed}/{total} ({(failed / total * 100.0):.1f}%)")

    y = pd.to_numeric(df[target_metric], errors="coerce")
    scored = df.assign(_target=y).dropna(subset=["_target"])
    if not scored.empty:
        best_idx = scored["_target"].idxmax() if higher_is_better else scored["_target"].idxmin()
        best_row = scored.loc[best_idx]
        insights.append(f"最良run: run_id={best_row.get('run_id')} {target_metric}={float(best_row['_target']):.6g}")

    if not impact_df.empty:
        top = impact_df.iloc[0]
        insights.append(
            "影響最大パラメータ: "
            f"{top['parameter']} effect_abs={float(top['effect_abs']):.6g} p={top.get('permutation_pvalue')}"
        )

    if not reflection_df.empty:
        worst = reflection_df.iloc[0]
        insights.append(
            "設定反映監査: "
            f"{worst['parameter']} mismatch_rate={float(worst['mismatch_rate']) * 100.0:.1f}% "
            f"(n={int(worst['compared_rows'])})"
        )

    if not contribution_df.empty:
        top_feature = contribution_df.iloc[0]
        insights.append(
            "精度寄与トップ: "
            f"{top_feature['feature']} contribution={float(top_feature['contribution_rate']) * 100.0:.1f}% "
            f"meta_model_r2={float(top_feature['model_r2']):.3f}"
        )

    if not causal_hints_df.empty:
        top_causal = causal_hints_df.iloc[0]
        insights.append(
            "因果候補(代理): "
            f"{top_causal['treatment']} std_coef={float(top_causal['std_coef']):.3f} "
            f"p={float(top_causal['pvalue']):.4g}"
        )

    return insights


def fetch_meta_automodel_results(
    engine: Engine,
    config_id: int | None = None,
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 5000,
) -> pd.DataFrame:
    schema = _safe_ident(settings.model_schema)
    table = _safe_ident(settings.model_table)
    where: list[str] = []
    params: dict[str, Any] = {"limit": int(max(1, limit))}
    if config_id is not None:
        where.append("config_id = :config_id")
        params["config_id"] = int(config_id)
    if run_id:
        where.append("run_id = :run_id")
        params["run_id"] = str(run_id)
    if status:
        where.append("status = :status")
        params["status"] = str(status)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = text(
        f"""
        SELECT *
        FROM {schema}.{table}
        {where_sql}
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def build_meta_automodel_report(
    raw_df: pd.DataFrame,
    target_metric: str = "metrics.mae",
    higher_is_better: bool = False,
    recursive_depth: int = 3,
    min_group_size: int = 5,
    alpha: float = 0.05,
    top_k: int = 20,
) -> dict[str, Any]:
    flat = _prepare_flattened_results(raw_df)
    resolved_target = _resolve_target_metric(flat, target_metric=target_metric)
    metric_summary_df = _summarize_metric_columns(flat)
    reflection_df, reflection_detail_df = _build_parameter_reflection(flat)

    status_counts = (
        flat["status"].astype(str).value_counts().to_dict() if "status" in flat.columns and not flat.empty else {}
    )
    overview = {
        "rows": int(flat.shape[0]),
        "columns": int(flat.shape[1]),
        "config_count": int(flat["config_id"].nunique()) if "config_id" in flat.columns and not flat.empty else 0,
        "run_count": int(flat["run_id"].nunique()) if "run_id" in flat.columns and not flat.empty else 0,
        "model_count": int(flat["model_name"].nunique()) if "model_name" in flat.columns and not flat.empty else 0,
        "status_counts": status_counts,
    }
    if "created_at" in flat.columns and not flat.empty:
        overview["created_at_min"] = (
            flat["created_at"].min().isoformat() if pd.notna(flat["created_at"].min()) else None
        )
        overview["created_at_max"] = (
            flat["created_at"].max().isoformat() if pd.notna(flat["created_at"].max()) else None
        )

    if resolved_target is None:
        return {
            "overview": overview,
            "target_metric": None,
            "metric_summary": metric_summary_df,
            "parameter_impact": pd.DataFrame(),
            "parameter_reflection": reflection_df,
            "parameter_reflection_detail": reflection_detail_df,
            "feature_contribution": pd.DataFrame(),
            "causal_hints": pd.DataFrame(),
            "stat_tests": {"error": "target_metric_not_found"},
            "recursive_tree": {"depth": 0, "rows": int(flat.shape[0]), "children": []},
            "flattened_results": flat,
            "insights": ["metrics.* 列が見つからないため検定と再帰分析をスキップしました。"],
        }

    impact_df = _parameter_impact(flat, target_metric=resolved_target, alpha=float(alpha))
    ranked_params = impact_df["parameter"].astype(str).tolist() if "parameter" in impact_df.columns else []
    recursive_tree = _recursive_drilldown(
        df=flat,
        target_metric=resolved_target,
        ranked_params=ranked_params,
        depth=0,
        max_depth=max(1, int(recursive_depth)),
        min_group_size=max(2, int(min_group_size)),
    )
    stat_tests = _build_stat_tests(flat, target_metric=resolved_target, alpha=float(alpha))
    contribution_df = _build_feature_contribution(
        flat,
        target_metric=resolved_target,
        max_features=max(20, int(top_k)),
        random_state=42,
    )
    causal_candidates = [
        c
        for c in (impact_df["parameter"].astype(str).tolist() if "parameter" in impact_df.columns else [])
        if c in flat.columns and pd.to_numeric(flat[c], errors="coerce").notna().sum() >= 30
    ]
    if not causal_candidates:
        causal_candidates = [
            c
            for c in flat.columns
            if c.startswith("params.") and pd.to_numeric(flat[c], errors="coerce").notna().sum() >= 30
        ]
    causal_hints_df = _build_causal_hints(
        flat,
        target_metric=resolved_target,
        candidate_cols=causal_candidates[:20],
        max_rows=max(10, min(40, int(top_k))),
    )
    insights = _build_key_insights(
        df=flat,
        target_metric=resolved_target,
        impact_df=impact_df,
        reflection_df=reflection_df,
        contribution_df=contribution_df,
        causal_hints_df=causal_hints_df,
        higher_is_better=bool(higher_is_better),
    )

    return {
        "overview": overview,
        "target_metric": resolved_target,
        "metric_summary": metric_summary_df,
        "parameter_impact": impact_df.head(max(1, int(top_k))).copy(),
        "parameter_reflection": reflection_df,
        "parameter_reflection_detail": reflection_detail_df,
        "feature_contribution": contribution_df,
        "causal_hints": causal_hints_df,
        "stat_tests": stat_tests,
        "recursive_tree": recursive_tree,
        "flattened_results": flat,
        "insights": insights,
    }


def generate_meta_automodel_report(
    config_id: int | None = None,
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 5000,
    target_metric: str = "metrics.mae",
    higher_is_better: bool = False,
    recursive_depth: int = 3,
    min_group_size: int = 5,
    alpha: float = 0.05,
    out_dir: str | Path | None = None,
    top_k: int = 20,
    write_outputs: bool = True,
) -> dict[str, Any]:
    engine = make_engine()
    raw_df = fetch_meta_automodel_results(
        engine=engine,
        config_id=config_id,
        run_id=run_id,
        status=status,
        limit=limit,
    )
    report = build_meta_automodel_report(
        raw_df=raw_df,
        target_metric=target_metric,
        higher_is_better=higher_is_better,
        recursive_depth=recursive_depth,
        min_group_size=min_group_size,
        alpha=alpha,
        top_k=top_k,
    )

    output_dir: Path | None = None
    files: dict[str, str] = {}
    if write_outputs:
        if out_dir is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = (settings.artifact_dir / "reports" / f"meta_automodel_{ts}").resolve()
        else:
            output_dir = Path(out_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        flat_df: pd.DataFrame = report["flattened_results"]
        metric_summary_df: pd.DataFrame = report["metric_summary"]
        impact_df: pd.DataFrame = report["parameter_impact"]
        reflection_df: pd.DataFrame = report.get("parameter_reflection", pd.DataFrame())
        reflection_detail_df: pd.DataFrame = report.get("parameter_reflection_detail", pd.DataFrame())
        contribution_df: pd.DataFrame = report.get("feature_contribution", pd.DataFrame())
        causal_hints_df: pd.DataFrame = report.get("causal_hints", pd.DataFrame())
        recursive_tree = report["recursive_tree"]

        flat_path = output_dir / "flattened_results.csv"
        metric_path = output_dir / "metric_summary.csv"
        impact_path = output_dir / "parameter_impact.csv"
        reflection_path = output_dir / "parameter_reflection.csv"
        reflection_detail_path = output_dir / "parameter_reflection_detail.csv"
        contribution_path = output_dir / "feature_contribution.csv"
        causal_hints_path = output_dir / "causal_hints.csv"
        report_path = output_dir / "report.json"
        tree_path = output_dir / "recursive_tree.json"

        flat_df.to_csv(flat_path, index=False)
        metric_summary_df.to_csv(metric_path, index=False)
        impact_df.to_csv(impact_path, index=False)
        reflection_df.to_csv(reflection_path, index=False)
        reflection_detail_df.to_csv(reflection_detail_path, index=False)
        contribution_df.to_csv(contribution_path, index=False)
        causal_hints_df.to_csv(causal_hints_path, index=False)
        tree_path.write_text(json.dumps(recursive_tree, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        target = report["target_metric"]
        if target is not None:
            _save_plot_status_counts(flat_df, output_dir / "status_counts.png")
            _save_plot_target_hist(flat_df, target_metric=target, out_path=output_dir / "target_distribution.png")
            _save_plot_target_over_time(flat_df, target_metric=target, out_path=output_dir / "target_over_time.png")
            _save_plot_top_param_effects(impact_df, out_path=output_dir / "top_param_effects.png", top_k=12)

        payload = {
            "overview": report["overview"],
            "target_metric": report["target_metric"],
            "stat_tests": report["stat_tests"],
            "insights": report["insights"],
            "top_parameter_impact": impact_df.head(20).to_dict(orient="records"),
            "parameter_reflection": reflection_df.head(20).to_dict(orient="records"),
            "feature_contribution": contribution_df.head(20).to_dict(orient="records"),
            "causal_hints": causal_hints_df.head(20).to_dict(orient="records"),
            "output_dir": str(output_dir),
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        files = {
            "report_json": str(report_path),
            "recursive_tree_json": str(tree_path),
            "flattened_csv": str(flat_path),
            "metric_summary_csv": str(metric_path),
            "parameter_impact_csv": str(impact_path),
            "parameter_reflection_csv": str(reflection_path),
            "parameter_reflection_detail_csv": str(reflection_detail_path),
            "feature_contribution_csv": str(contribution_path),
            "causal_hints_csv": str(causal_hints_path),
            "status_counts_png": str(output_dir / "status_counts.png"),
            "target_distribution_png": str(output_dir / "target_distribution.png"),
            "target_over_time_png": str(output_dir / "target_over_time.png"),
            "top_param_effects_png": str(output_dir / "top_param_effects.png"),
        }

    return {
        "ok": True,
        "rows": int(raw_df.shape[0]),
        "target_metric": report["target_metric"],
        "overview": report["overview"],
        "insights": report["insights"],
        "stat_tests": report["stat_tests"],
        "top_parameter_impact": report["parameter_impact"].head(max(1, int(top_k))).to_dict(orient="records"),
        "parameter_reflection": report.get("parameter_reflection", pd.DataFrame())
        .head(max(1, int(top_k)))
        .to_dict(orient="records"),
        "feature_contribution": report.get("feature_contribution", pd.DataFrame())
        .head(max(1, int(top_k)))
        .to_dict(orient="records"),
        "causal_hints": report.get("causal_hints", pd.DataFrame()).head(max(1, int(top_k))).to_dict(orient="records"),
        "recursive_tree": report["recursive_tree"],
        "output_dir": str(output_dir) if output_dir is not None else None,
        "files": files,
    }
