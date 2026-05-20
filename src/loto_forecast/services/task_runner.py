from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import ray

from loto_forecast.analysis.forecast_analysis import (
    analyze_exogenous,
    build_explainability_contract,
    build_relation_graph,
    compute_drift_metrics,
    ljung_box_test,
    plot_actual_vs_pred,
)
from loto_forecast.infra.db import get_session, init_db
from loto_forecast.infra.orm_models import (
    Evaluation,
    Model,
    PredictionRow,
    ResourceSample,
    Task,
)
from loto_forecast.services.resource_logger import start_resource_logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def load_callable(path: str) -> Callable[..., Any]:
    if ":" not in str(path):
        raise ValueError(f"callable must be like 'pkg.mod:func', got: {path}")
    mod, fn = str(path).split(":", 1)
    m = importlib.import_module(mod)
    f = getattr(m, fn)
    if not callable(f):
        raise TypeError(f"{path} is not callable")
    return f


def ensure_ray() -> None:
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    except Exception:
        ray.init(ignore_reinit_error=True, log_to_driver=False)


def _resource_summary(task_id: str) -> dict[str, float]:
    with get_session() as s:
        rows = s.query(ResourceSample).filter(ResourceSample.task_id == task_id).order_by(ResourceSample.ts.asc()).all()
    if not rows:
        return {}

    def _avg(name: str) -> float:
        vals = [float(getattr(r, name)) for r in rows if getattr(r, name) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "cpu_percent_avg": _avg("cpu_percent"),
        "rss_mb_avg": _avg("rss_mb"),
        "vms_mb_avg": _avg("vms_mb"),
        "gpu_util_avg": _avg("gpu_util"),
        "gpu_mem_mb_avg": _avg("gpu_mem_mb"),
        "gpu_temp_c_avg": _avg("gpu_temp_c"),
        "sample_count": float(len(rows)),
    }


def _latest_eval_predictions(dataset_id: str | None) -> dict[str, Any] | None:
    if not dataset_id:
        return None
    with get_session() as s:
        ev = (
            s.query(Evaluation)
            .filter(Evaluation.dataset_id == str(dataset_id))
            .order_by(Evaluation.created_at.desc(), Evaluation.id.desc())
            .first()
        )
        if ev is None:
            return None
        rows = (
            s.query(PredictionRow)
            .filter(PredictionRow.evaluation_id == int(ev.id))
            .order_by(PredictionRow.id.asc())
            .all()
        )
    if not rows:
        return {
            "evaluation_id": int(ev.id),
            "created_at": ev.created_at.isoformat() if ev.created_at is not None else None,
            "y_true": [],
            "y_pred": [],
            "residual": [],
        }
    y_true = np.asarray([float(r.y_true) for r in rows], dtype=float).reshape(-1)
    y_pred = np.asarray([float(r.y_pred) for r in rows], dtype=float).reshape(-1)
    residual = y_true - y_pred
    return {
        "evaluation_id": int(ev.id),
        "created_at": ev.created_at.isoformat() if ev.created_at is not None else None,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "residual": residual.tolist(),
    }


@ray.remote
def run_pipeline_task(task_id: str, callable_path: str, params: dict[str, Any]) -> None:
    init_db()
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            return
        t.status = "running"
        t.started_at = utcnow()
        s.commit()

    stop_event, th = start_resource_logger(task_id, interval_s=float(params.get("resource_interval_s", 1.0)))

    try:
        fn = load_callable(callable_path)
        out = fn(params)
        if not isinstance(out, dict):
            raise TypeError("callable output must be dict")

        model_info = dict(out.get("model", {}) or {})
        eval_info = dict(out.get("evaluation", {}) or {})
        pred = dict(out.get("pred", {}) or {})
        dataset_id = eval_info.get("dataset_id")

        y_true = np.asarray(pred.get("y_true", []), dtype=float).reshape(-1)
        y_pred = np.asarray(pred.get("y_pred", []), dtype=float).reshape(-1)
        t_index = list(pred.get("t", []))
        n = int(min(len(y_true), len(y_pred)))
        if len(t_index) < n:
            t_index = [None] * n

        residuals = (y_true[:n] - y_pred[:n]) if n > 0 else np.array([])
        lb = ljung_box_test(residuals, lags=int(params.get("ljung_box_lags", 10)))
        metrics = dict(eval_info.get("metrics", {}) or {})
        if n > 0:
            metrics.setdefault("mae", float(np.mean(np.abs(residuals))))
            metrics.setdefault("rmse", float(np.sqrt(np.mean(residuals**2))))
            if np.any(y_true[:n] != 0):
                metrics.setdefault(
                    "mape",
                    float(
                        np.nanmean(np.abs((y_true[:n] - y_pred[:n]) / np.where(y_true[:n] == 0, np.nan, y_true[:n])))
                        * 100.0
                    ),
                )

        # optional exogenous analysis: expects X_exog + y in output or params
        exog_analysis: dict[str, Any] = {}
        X_exog = out.get("X_exog", params.get("X_exog"))
        y_for_exog = out.get("y_exog_target", params.get("y_exog_target", y_true[:n].tolist() if n > 0 else []))
        if X_exog is not None:
            try:
                import pandas as pd

                X_df = pd.DataFrame(X_exog)
                y_series = pd.Series(y_for_exog)
                exog_res = analyze_exogenous(X_df, y_series)
                exog_analysis = {
                    "correlations": exog_res.correlations,
                    "spearman": exog_res.spearman,
                    "mutual_info": exog_res.mutual_info,
                    "lag_corr": exog_res.lag_corr,
                    "permutation": exog_res.permutation,
                    "shap_mean_abs": exog_res.shap_mean_abs,
                }
            except Exception as e:
                exog_analysis = {"error": f"exog_analysis_failed: {e}"}

        explain_contract = build_explainability_contract(
            y_true=y_true[:n],
            y_pred=y_pred[:n],
            exog_analysis=exog_analysis,
            X_exog=X_exog,
            what_if_scenarios=params.get("what_if_scenarios"),
            interval_coverage=float(params.get("interval_coverage", 0.9)),
            attribution_top_k=int(params.get("attribution_top_k", 5)),
            residual_lags=int(params.get("ljung_box_lags", 10)),
        )

        drift_analysis: dict[str, Any] = {"ok": True}
        prev_eval = _latest_eval_predictions(dataset_id)
        if prev_eval is None:
            drift_analysis.update({"has_reference": False, "reason": "no_previous_evaluation"})
        else:
            prev_y_true = np.asarray(prev_eval.get("y_true", []), dtype=float).reshape(-1)
            prev_y_pred = np.asarray(prev_eval.get("y_pred", []), dtype=float).reshape(-1)
            prev_res = np.asarray(prev_eval.get("residual", []), dtype=float).reshape(-1)
            cur_res = (y_true[:n] - y_pred[:n]) if n > 0 else np.array([])
            drift_analysis.update(
                {
                    "has_reference": True,
                    "reference_evaluation_id": prev_eval.get("evaluation_id"),
                    "reference_created_at": prev_eval.get("created_at"),
                    "y_true": compute_drift_metrics(prev_y_true, y_true[:n], bins=int(params.get("drift_bins", 10))),
                    "y_pred": compute_drift_metrics(prev_y_pred, y_pred[:n], bins=int(params.get("drift_bins", 10))),
                    "residual": compute_drift_metrics(prev_res, cur_res, bins=int(params.get("drift_bins", 10))),
                }
            )

        plot_path = plot_actual_vs_pred(
            y_true[:n],
            y_pred[:n],
            str(PROJECT_ROOT / "artifacts" / "plots" / f"task_{task_id}.png"),
            title=f"task={task_id} actual vs pred",
        )

        resource_summary = _resource_summary(task_id)
        relation_graph = build_relation_graph(
            model_name=str(model_info.get("name", "unknown")),
            metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            exog_scores=(
                {k: float(v) for k, v in exog_analysis.get("mutual_info", {}).items() if isinstance(v, (int, float))}
                if isinstance(exog_analysis, dict)
                else {}
            ),
            resource_summary={k: float(v) for k, v in resource_summary.items() if isinstance(v, (int, float))},
        )

        with get_session() as s:
            m = Model(
                name=str(model_info.get("name", "unknown")),
                version=model_info.get("version"),
                family=model_info.get("family"),
                properties=dict(model_info.get("properties", {}) or {}),
                hyperparams=dict(model_info.get("hyperparams", {}) or {}),
            )
            s.add(m)
            s.flush()

            ev = Evaluation(
                model_id=m.id,
                dataset_id=dataset_id,
                metrics=metrics,
                notes=eval_info.get("notes"),
                artifacts={
                    **dict(out.get("artifacts", {}) or {}),
                    "plot_actual_vs_pred_png": plot_path,
                },
                analysis={
                    "ljung_box": lb,
                    "exogenous": exog_analysis,
                    "explainability_contract": explain_contract,
                    "drift": drift_analysis,
                    "resource_summary": resource_summary,
                    "relation_graph": relation_graph,
                },
            )
            s.add(ev)
            s.flush()
            eval_id = int(ev.id)

            for i in range(n):
                s.add(
                    PredictionRow(
                        evaluation_id=eval_id,
                        t=None if t_index[i] is None else str(t_index[i]),
                        y_true=float(y_true[i]),
                        y_pred=float(y_pred[i]),
                    )
                )
            s.commit()

        with get_session() as s:
            t = s.get(Task, task_id)
            if t is not None:
                t.status = "succeeded"
                t.finished_at = utcnow()
                t.result = {
                    "evaluation_id": eval_id,
                    "model": model_info,
                    "evaluation": {"dataset_id": eval_info.get("dataset_id"), "metrics": metrics},
                    "artifacts": {"plot_actual_vs_pred_png": plot_path},
                    "contract": {
                        "attribution_top_features": explain_contract.get("attribution", {}).get("top_features", []),
                        "prediction_interval": explain_contract.get("prediction_interval", {}),
                    },
                }
                s.commit()

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        with get_session() as s:
            t = s.get(Task, task_id)
            if t is not None:
                t.status = "failed"
                t.finished_at = utcnow()
                t.error = err
                s.commit()
    finally:
        stop_event.set()
        with contextlib.suppress(Exception):
            th.join(timeout=2.0)


def submit_task(
    kind: str,
    callable_path: str,
    params: dict[str, Any] | None = None,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
) -> str:
    init_db()
    ensure_ray()
    task_id = str(uuid.uuid4())
    payload = dict(params or {})
    with get_session() as s:
        s.add(
            Task(
                id=task_id,
                kind=str(kind),
                status="queued",
                params={
                    "callable": callable_path,
                    "params": payload,
                    "num_cpus": float(num_cpus),
                    "num_gpus": float(num_gpus),
                },
            )
        )
        s.commit()

    run_pipeline_task.options(num_cpus=float(num_cpus), num_gpus=float(num_gpus)).remote(
        task_id,
        callable_path,
        payload,
    )
    return task_id


def submit_recursive_tasks(
    kind: str,
    callable_path: str,
    params: dict[str, Any] | None = None,
    recursive_depth: int = 3,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
    strategy: str = "seed_increment",
    seed_key: str = "seed",
    seed_start: int | None = None,
    seed_step: int = 1,
) -> dict[str, Any]:
    depth = max(1, int(recursive_depth))
    loop_id = str(uuid.uuid4())
    base = dict(params or {})
    base_seed = int(seed_start) if seed_start is not None else int(base.get(seed_key, 1))
    step = max(1, int(seed_step))
    ids: list[str] = []

    for i in range(depth):
        p = dict(base)
        p["loop_id"] = loop_id
        p["loop_iteration"] = int(i + 1)
        p["loop_depth"] = int(depth)
        p["loop_strategy"] = str(strategy)
        if strategy == "seed_increment":
            p[seed_key] = int(base_seed + i * step)
        ids.append(
            submit_task(
                kind=kind,
                callable_path=callable_path,
                params=p,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )
        )

    return {
        "loop_id": loop_id,
        "recursive_depth": depth,
        "strategy": strategy,
        "task_ids": ids,
    }
