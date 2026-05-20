from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from scipy import stats as spstats

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:
    acorr_ljungbox = None


@dataclass
class ExogAnalysisResult:
    correlations: dict[str, float]
    spearman: dict[str, float]
    mutual_info: dict[str, float]
    lag_corr: dict[str, dict[int, float]]
    permutation: dict[str, float] | None = None
    shap_mean_abs: dict[str, float] | None = None


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _to_float_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def plot_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png_path: str,
    title: str = "Actual vs Predicted",
) -> str:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = int(min(len(y_true), len(y_pred)))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    path = Path(out_png_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="actual")
    plt.plot(y_pred, label="pred")
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def build_conformal_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage: float = 0.9,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    n = int(min(y.shape[0], p.shape[0]))
    if n <= 0:
        return {"ok": False, "reason": "empty"}
    y = y[:n]
    p = p[:n]
    residual_abs = np.abs(y - p)
    if residual_abs.size == 0:
        return {"ok": False, "reason": "empty"}

    c = float(min(max(coverage, 0.5), 0.999))
    q = float(np.quantile(residual_abs, c, method="higher"))
    lower = p - q
    upper = p + q
    emp = float(np.mean((y >= lower) & (y <= upper)))
    width = upper - lower
    return {
        "ok": True,
        "coverage_target": c,
        "quantile_abs_residual": q,
        "empirical_coverage": emp,
        "interval_width_mean": float(np.mean(width)),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }


def _population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float | None:
    ref = _to_float_array(reference)
    cur = _to_float_array(current)
    if ref.size < 20 or cur.size < 20:
        return None
    b = int(max(4, min(50, bins)))
    try:
        edges = np.quantile(ref, np.linspace(0.0, 1.0, b + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            return None
        eps = 1e-8
        ref_hist, _ = np.histogram(ref, bins=edges)
        cur_hist, _ = np.histogram(cur, bins=edges)
        ref_ratio = np.maximum(ref_hist / max(ref_hist.sum(), 1), eps)
        cur_ratio = np.maximum(cur_hist / max(cur_hist.sum(), 1), eps)
        psi = np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio))
        return float(psi)
    except Exception:
        return None


def compute_drift_metrics(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    ref = _to_float_array(reference)
    cur = _to_float_array(current)
    out: dict[str, Any] = {
        "ok": bool(ref.size > 0 and cur.size > 0),
        "reference_n": int(ref.size),
        "current_n": int(cur.size),
        "reference_mean": _safe_float(np.mean(ref)) if ref.size else None,
        "current_mean": _safe_float(np.mean(cur)) if cur.size else None,
        "mean_shift": _safe_float(np.mean(cur) - np.mean(ref)) if ref.size and cur.size else None,
        "psi": _population_stability_index(ref, cur, bins=bins),
    }
    if SCIPY_AVAILABLE and ref.size >= 20 and cur.size >= 20:
        try:
            ks = spstats.ks_2samp(ref, cur, alternative="two-sided", method="auto")
            out["ks_stat"] = _safe_float(ks.statistic)
            out["ks_pvalue"] = _safe_float(ks.pvalue)
        except Exception:
            out["ks_stat"] = None
            out["ks_pvalue"] = None
    else:
        out["ks_stat"] = None
        out["ks_pvalue"] = None
    return out


def _safe_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def summarize_attributions(exog_analysis: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    if not isinstance(exog_analysis, dict) or not exog_analysis:
        return []

    score_sources = ["shap_mean_abs", "permutation", "mutual_info", "correlations"]
    source_name = ""
    raw_scores: dict[str, Any] = {}
    for key in score_sources:
        val = exog_analysis.get(key)
        if isinstance(val, dict) and val:
            raw_scores = val
            source_name = key
            break
    if not raw_scores:
        return []

    corr = exog_analysis.get("correlations", {})
    lag_corr = exog_analysis.get("lag_corr", {})
    rows: list[dict[str, Any]] = []
    for feat, score in raw_scores.items():
        s = _safe_float(abs(score) if source_name == "correlations" else score)
        if s is None:
            continue
        direction = None
        if isinstance(corr, dict) and feat in corr:
            c = _safe_float(corr.get(feat))
            if c is not None:
                direction = "positive" if c >= 0 else "negative"
        best_lag = None
        if isinstance(lag_corr, dict) and isinstance(lag_corr.get(feat), dict):
            lag_vals: dict[int, float] = {}
            for k, v in lag_corr.get(feat, {}).items():
                try:
                    lag_key = int(k)
                    lag_value = abs(float(v))
                except (TypeError, ValueError):
                    lag_key = None
                if lag_key is not None:
                    lag_vals[lag_key] = lag_value
            if lag_vals:
                best_lag = max(lag_vals.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "feature": str(feat),
                "score": float(s),
                "source": source_name,
                "direction": direction,
                "best_lag": best_lag,
            }
        )
    rows = sorted(rows, key=lambda x: float(x["score"]), reverse=True)
    return rows[: max(1, int(top_k))]


def run_what_if_scenarios(
    X: pd.DataFrame | dict[str, Any] | None,
    y_pred: np.ndarray,
    scenarios: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if X is None:
        return []
    try:
        Xf = _safe_numeric_df(pd.DataFrame(X))
    except Exception:
        return []
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = int(min(Xf.shape[0], yp.shape[0]))
    if n <= 2:
        return []
    Xf = Xf.iloc[:n].copy()
    yp = yp[:n]

    if not scenarios:
        # default scenarios: top numeric columns by variance
        variances = Xf.var(numeric_only=True).sort_values(ascending=False).head(3).index.tolist()
        scenarios = [{"name": f"{c}+5%", "feature": c, "pct": 0.05} for c in variances]

    out: list[dict[str, Any]] = []
    for sc in scenarios:
        feat = str(sc.get("feature", "")).strip()
        if feat == "" or feat not in Xf.columns:
            continue
        xs = pd.to_numeric(Xf[feat], errors="coerce")
        valid = xs.notna() & np.isfinite(yp)
        if valid.sum() < 3:
            continue
        xv = xs.loc[valid].to_numpy(dtype=float)
        yv = yp[valid.to_numpy()]
        if np.nanstd(xv) <= 0:
            continue
        slope = float(np.polyfit(xv, yv, deg=1)[0])
        baseline = float(np.nanmean(xv))
        if "delta" in sc:
            delta_x = float(sc["delta"])
        elif "multiplier" in sc:
            delta_x = baseline * (float(sc["multiplier"]) - 1.0)
        else:
            delta_x = baseline * float(sc.get("pct", 0.0))
        delta_y = slope * delta_x
        out.append(
            {
                "name": str(sc.get("name", f"{feat}_scenario")),
                "feature": feat,
                "delta_x": float(delta_x),
                "estimated_delta_y": float(delta_y),
                "slope": slope,
            }
        )
    return out


def analyze_exogenous(
    X: pd.DataFrame,
    y: pd.Series,
    model_for_perm: Any | None = None,
    model_for_shap: Any | None = None,
    n_repeats: int = 10,
    max_lag: int = 7,
    random_state: int = 42,
) -> ExogAnalysisResult:
    yv = pd.to_numeric(pd.Series(y), errors="coerce")
    Xn = _safe_numeric_df(X)
    valid = yv.notna()
    Xn = Xn.loc[valid]
    yv = yv.loc[valid]
    Xf = Xn.fillna(Xn.median(numeric_only=True))

    pearson: dict[str, float] = {}
    spearman: dict[str, float] = {}
    lag_corr: dict[str, dict[int, float]] = {}
    for c in Xf.columns:
        xs = Xf[c]
        pearson[c] = float(xs.corr(yv, method="pearson"))
        spearman[c] = float(xs.corr(yv, method="spearman"))
        lag_corr[c] = {}
        for lag in range(0, max(0, int(max_lag)) + 1):
            try:
                lag_corr[c][lag] = float(xs.shift(lag).corr(yv))
            except Exception:
                lag_corr[c][lag] = float("nan")

    if Xf.shape[0] >= 5 and Xf.shape[1] >= 1:
        mi = mutual_info_regression(Xf.values, yv.values, random_state=random_state)
        mi_map = {c: float(v) for c, v in zip(Xf.columns, mi, strict=False)}
    else:
        mi_map = {c: float("nan") for c in Xf.columns}

    perm_map = None
    if model_for_perm is not None and Xf.shape[0] >= 10:
        try:
            r = permutation_importance(
                model_for_perm,
                Xf.values,
                yv.values,
                n_repeats=int(max(3, n_repeats)),
                random_state=random_state,
            )
            perm_map = {c: float(v) for c, v in zip(Xf.columns, r.importances_mean, strict=False)}
        except Exception:
            perm_map = None

    shap_map = None
    if SHAP_AVAILABLE and model_for_shap is not None and Xf.shape[0] >= 10:
        try:
            sample_n = int(min(256, Xf.shape[0]))
            Xs = Xf.sample(sample_n, random_state=random_state)
            explainer = shap.Explainer(model_for_shap, Xs)
            sv = explainer(Xs)
            vals = np.asarray(sv.values)
            if vals.ndim == 3:
                vals = vals[:, :, 0]
            mean_abs = np.mean(np.abs(vals), axis=0)
            shap_map = {c: float(v) for c, v in zip(Xs.columns, mean_abs, strict=False)}
        except Exception:
            shap_map = None

    return ExogAnalysisResult(
        correlations=pearson,
        spearman=spearman,
        mutual_info=mi_map,
        lag_corr=lag_corr,
        permutation=perm_map,
        shap_mean_abs=shap_map,
    )


def build_explainability_contract(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exog_analysis: dict[str, Any] | None = None,
    X_exog: pd.DataFrame | dict[str, Any] | None = None,
    what_if_scenarios: list[dict[str, Any]] | None = None,
    interval_coverage: float = 0.9,
    attribution_top_k: int = 5,
    residual_lags: int = 10,
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = int(min(yt.shape[0], yp.shape[0]))
    yt = yt[:n]
    yp = yp[:n]
    residuals = yt - yp if n > 0 else np.array([])

    interval = build_conformal_interval(yt, yp, coverage=interval_coverage)
    lb = ljung_box_test(residuals, lags=int(max(2, residual_lags)))
    attrs = summarize_attributions(exog_analysis or {}, top_k=attribution_top_k)
    what_if = run_what_if_scenarios(X_exog, yp, scenarios=what_if_scenarios)

    return {
        "point_forecast": {
            "count": int(n),
            "mean": _safe_float(np.mean(yp)) if n else None,
            "std": _safe_float(np.std(yp, ddof=1)) if n > 1 else 0.0,
            "sample": [float(v) for v in yp[: min(20, n)]],
        },
        "prediction_interval": {
            "ok": bool(interval.get("ok")),
            "coverage_target": interval.get("coverage_target"),
            "quantile_abs_residual": interval.get("quantile_abs_residual"),
            "empirical_coverage": interval.get("empirical_coverage"),
            "interval_width_mean": interval.get("interval_width_mean"),
            "lower_sample": list(interval.get("lower", [])[:20]),
            "upper_sample": list(interval.get("upper", [])[:20]),
        },
        "attribution": {
            "top_features": attrs,
            "sources_available": sorted(list((exog_analysis or {}).keys())),
        },
        "what_if": {"scenarios": what_if},
        "residual_diagnostics": {
            "mean": _safe_float(np.mean(residuals)) if residuals.size else None,
            "std": _safe_float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0,
            "mae": _safe_float(np.mean(np.abs(residuals))) if residuals.size else None,
            "rmse": _safe_float(np.sqrt(np.mean(residuals**2))) if residuals.size else None,
            "ljung_box": lb,
        },
    }


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> dict[str, Any]:
    if acorr_ljungbox is None:
        return {"ok": False, "reason": "statsmodels_missing"}
    r = np.asarray(residuals).reshape(-1)
    if r.shape[0] < max(12, lags + 2):
        return {"ok": False, "reason": "insufficient_data", "n": int(r.shape[0])}
    df = acorr_ljungbox(r, lags=[int(lags)], return_df=True)
    return {
        "ok": True,
        "lags": int(lags),
        "lb_stat": float(df["lb_stat"].iloc[-1]),
        "lb_pvalue": float(df["lb_pvalue"].iloc[-1]),
    }


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    power: int = 2,
) -> dict[str, Any]:
    if not SCIPY_AVAILABLE:
        return {"ok": False, "reason": "scipy_missing"}
    y = np.asarray(y_true).reshape(-1)
    p1 = np.asarray(y_pred_1).reshape(-1)
    p2 = np.asarray(y_pred_2).reshape(-1)
    n = int(min(y.shape[0], p1.shape[0], p2.shape[0]))
    if n < 10:
        return {"ok": False, "reason": "insufficient_data", "n": n}
    y = y[:n]
    p1 = p1[:n]
    p2 = p2[:n]
    e1 = np.abs(y - p1) ** int(power)
    e2 = np.abs(y - p2) ** int(power)
    d = e1 - e2
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d <= 0:
        return {"ok": False, "reason": "zero_variance"}
    dm_stat = mean_d / math.sqrt(var_d / n)
    pval = 2.0 * (1.0 - spstats.norm.cdf(abs(dm_stat)))
    return {"ok": True, "n": n, "dm_stat": float(dm_stat), "pvalue": float(pval), "mean_loss_diff": mean_d}


def build_relation_graph(
    model_name: str,
    metrics: dict[str, float],
    exog_scores: dict[str, float],
    resource_summary: dict[str, float],
) -> dict[str, Any]:
    g = nx.Graph()
    g.add_node(f"model:{model_name}", kind="model")
    for k, v in metrics.items():
        mk = f"metric:{k}"
        g.add_node(mk, kind="metric", weight=float(v))
        g.add_edge(f"model:{model_name}", mk, weight=abs(float(v)))
    for k, v in exog_scores.items():
        ek = f"exog:{k}"
        g.add_node(ek, kind="exog", weight=float(v))
        g.add_edge(f"model:{model_name}", ek, weight=abs(float(v)))
    for k, v in resource_summary.items():
        rk = f"res:{k}"
        g.add_node(rk, kind="resource", weight=float(v))
        g.add_edge(f"model:{model_name}", rk, weight=abs(float(v)))

    deg = nx.degree_centrality(g)
    btw = nx.betweenness_centrality(g)
    cen = []
    for n, d in deg.items():
        cen.append({"node": n, "degree_centrality": float(d), "betweenness": float(btw.get(n, 0.0))})
    cen = sorted(cen, key=lambda x: x["betweenness"], reverse=True)
    return {
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"source": u, "target": v, **g.edges[(u, v)]} for u, v in g.edges],
        "centrality": cen,
    }
