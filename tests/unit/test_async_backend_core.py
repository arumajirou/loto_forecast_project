from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from loto_forecast.analysis.forecast_analysis import (
    analyze_exogenous,
    build_conformal_interval,
    build_explainability_contract,
    compute_drift_metrics,
    ljung_box_test,
    plot_actual_vs_pred,
)
from loto_forecast.infra.db import get_session, init_db
from loto_forecast.infra.orm_models import Task
from loto_forecast.pipeline_hooks import demo_train_and_predict


def test_demo_hook_and_plot(tmp_path: Path) -> None:
    out = demo_train_and_predict({"n": 64, "seed": 1})
    assert "pred" in out
    y_true = np.asarray(out["pred"]["y_true"], dtype=float)
    y_pred = np.asarray(out["pred"]["y_pred"], dtype=float)
    png = plot_actual_vs_pred(y_true, y_pred, str(tmp_path / "avp.png"))
    assert Path(png).exists()


def test_exog_analysis_smoke() -> None:
    n = 80
    x = np.linspace(0, 4, n)
    X = pd.DataFrame({"a": np.sin(x), "b": np.cos(x), "c": np.random.default_rng(1).normal(size=n)})
    y = pd.Series(np.sin(x + 0.1))
    res = analyze_exogenous(X, y, model_for_perm=None, model_for_shap=None, max_lag=3)
    assert "a" in res.correlations
    assert "a" in res.mutual_info
    assert 0 in res.lag_corr["a"]


def test_ljung_box_and_sqlite_registry() -> None:
    init_db()
    lb = ljung_box_test(np.random.default_rng(0).normal(size=120), lags=8)
    assert "ok" in lb
    task_id = f"test-task-{uuid.uuid4()}"
    with get_session() as s:
        s.add(Task(id=task_id, kind="train", status="queued", params={}))
        s.commit()
        t = s.get(Task, task_id)
        assert t is not None


def test_conformal_contract_and_drift() -> None:
    rng = np.random.default_rng(7)
    n = 120
    x = np.linspace(0.0, 8.0, n)
    y_true = np.sin(x) + 0.05 * rng.normal(size=n)
    y_pred = np.sin(x + 0.1)
    X = pd.DataFrame({"feat_a": np.cos(x), "feat_b": np.sin(0.5 * x)})
    exog = analyze_exogenous(X, pd.Series(y_true), max_lag=3)
    exog_dict = {
        "correlations": exog.correlations,
        "spearman": exog.spearman,
        "mutual_info": exog.mutual_info,
        "lag_corr": exog.lag_corr,
    }

    iv = build_conformal_interval(y_true, y_pred, coverage=0.9)
    assert iv["ok"] is True
    assert len(iv["lower"]) == n
    assert len(iv["upper"]) == n

    contract = build_explainability_contract(
        y_true=y_true,
        y_pred=y_pred,
        exog_analysis=exog_dict,
        X_exog=X,
        what_if_scenarios=[{"name": "feat_a+5%", "feature": "feat_a", "pct": 0.05}],
        interval_coverage=0.9,
        attribution_top_k=3,
    )
    assert "prediction_interval" in contract
    assert "attribution" in contract
    assert len(contract["attribution"]["top_features"]) >= 1
    assert "residual_diagnostics" in contract

    drift = compute_drift_metrics(y_true[:80], y_true[40:], bins=8)
    assert drift["ok"] is True
    assert drift["reference_n"] > 0
    assert drift["current_n"] > 0
