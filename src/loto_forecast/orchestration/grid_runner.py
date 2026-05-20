from __future__ import annotations

import itertools
import json
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any

from loguru import logger

from ..data.db import make_engine
from ..infra.logging_utils import setup_logging
from ..infra.meta_store import (
    create_grid_definition,
    finish_grid_task,
    get_grid_definition,
    list_grid_tasks,
    log_execution_event,
    mark_model_run_end,
    replace_grid_tasks,
    start_grid_task,
    upsert_model_run,
    write_resource_samples,
)
from ..infra.monitoring import sample_resources


def _normalize_param_space(param_space: dict[str, Any]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for key, value in param_space.items():
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


def expand_param_grid(param_space: dict[str, Any], max_tasks: int | None = None) -> list[dict[str, Any]]:
    norm = _normalize_param_space(param_space)
    if not norm:
        return [{}]

    keys = sorted(norm.keys())
    products = itertools.product(*[norm[k] for k in keys])

    out: list[dict[str, Any]] = []
    for tup in products:
        combo = {k: tup[i] for i, k in enumerate(keys)}
        out.append(combo)
        if max_tasks is not None and len(out) >= max_tasks:
            break
    return out


def create_grid(
    grid_id: str,
    library_name: str,
    adapter_name: str,
    model_name: str,
    horizon: int,
    param_space: dict[str, Any],
    exog_policy: dict[str, Any] | None = None,
    run_predict: bool = True,
    run_evaluate: bool = True,
    max_tasks: int | None = None,
    note: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    engine = make_engine()

    combos = expand_param_grid(param_space, max_tasks=max_tasks)
    create_grid_definition(
        engine,
        grid_id=grid_id,
        library_name=library_name,
        adapter_name=adapter_name,
        model_name=model_name,
        horizon=horizon,
        param_space=param_space,
        exog_policy=exog_policy or {},
        run_predict=run_predict,
        run_evaluate=run_evaluate,
        max_tasks=max_tasks,
        note=note,
        created_by=created_by,
    )
    replace_grid_tasks(engine, grid_id=grid_id, tasks=combos)

    return {
        "grid_id": grid_id,
        "task_count": len(combos),
        "model_name": model_name,
        "adapter_name": adapter_name,
    }


def _resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    cpu = [s["cpu_percent"] for s in samples]
    mem = [s["mem_percent"] for s in samples]
    rss = [s["rss_mb"] for s in samples]
    return {
        "sample_count": len(samples),
        "cpu_max": max(cpu),
        "cpu_mean": float(sum(cpu) / len(cpu)),
        "mem_max": max(mem),
        "mem_mean": float(sum(mem) / len(mem)),
        "rss_max_mb": max(rss),
        "rss_mean_mb": float(sum(rss) / len(rss)),
    }


def run_grid(grid_id: str, stop_on_error: bool = False) -> dict[str, Any]:
    from ..models.registry import get_adapter

    engine = make_engine()
    grid = get_grid_definition(engine, grid_id)
    if grid is None:
        raise ValueError(f"grid not found: {grid_id}")

    adapter_name = str(grid["adapter_name"])
    model_name = str(grid["model_name"])
    horizon = int(grid["horizon"])
    run_predict = bool(grid["run_predict"])
    run_evaluate = bool(grid["run_evaluate"])

    adapter = get_adapter(adapter_name)

    pending_tasks = list_grid_tasks(engine, grid_id, status="pending", limit=100000)
    if not pending_tasks:
        return {"grid_id": grid_id, "message": "no pending tasks", "executed": 0}

    success = 0
    failed = 0

    for t in pending_tasks:
        task_id = int(t["task_id"])
        task_order = int(t["task_order"])
        params = t["param_values"]
        if isinstance(params, str):
            params = json.loads(params)

        run_id = f"{grid_id}_t{task_order}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logfile = setup_logging(run_id)
        start_grid_task(engine, task_id=task_id, run_id=run_id, log_path=str(logfile))
        upsert_model_run(
            engine,
            run_id=run_id,
            model_name=model_name,
            meta={"grid_id": grid_id, "task_id": task_id, "params": params},
            library_name=str(grid["library_name"]),
            adapter_name=adapter_name,
            status="running",
            grid_id=grid_id,
            task_id=task_id,
            log_path=str(logfile),
        )
        log_execution_event(
            engine,
            task_id=task_id,
            run_id=run_id,
            event_type="grid_task_start",
            message=f"grid task started. model={model_name} params={params}",
            payload={"grid_id": grid_id, "task_order": task_order, "params": params},
        )

        res_samples: list[dict[str, Any]] = [asdict(sample_resources())]

        try:
            validation = adapter.validate(model_name=model_name, model_params=dict(params))
            if not validation.get("ok", False):
                raise ValueError(f"invalid params: {validation.get('errors', [])}")

            out = adapter.run(
                model_name=model_name,
                horizon=horizon,
                model_params=dict(params),
                run_predict=run_predict,
                run_evaluate=run_evaluate,
                run_id=run_id,
                grid_id=grid_id,
                task_id=task_id,
            )
            res_samples.append(asdict(sample_resources()))
            write_resource_samples(engine, run_id=out.run_id, samples=res_samples)

            result = {
                "run_id": out.run_id,
                "train": out.train,
                "predict": out.predict,
                "evaluate": out.evaluate,
                "validation": validation,
            }
            metrics = (out.evaluate or {}).get("metrics", {}) if out.evaluate else {}
            finish_grid_task(
                engine,
                task_id=task_id,
                status="success",
                result=result,
                metrics=metrics,
                resource_summary=_resource_summary(res_samples),
                error_message=None,
            )
            log_execution_event(
                engine,
                task_id=task_id,
                run_id=out.run_id,
                event_type="grid_task_success",
                message="grid task completed successfully",
                payload={"metrics": metrics},
            )
            success += 1
        except Exception as e:
            res_samples.append(asdict(sample_resources()))
            write_resource_samples(engine, run_id=run_id, samples=res_samples)
            err = f"{e}\n{traceback.format_exc()}"
            logger.error(err)
            mark_model_run_end(engine, run_id=run_id, status="failed", error_message=str(e))
            finish_grid_task(
                engine,
                task_id=task_id,
                status="failed",
                result={"run_id": run_id},
                metrics={},
                resource_summary=_resource_summary(res_samples),
                error_message=str(e),
            )
            log_execution_event(
                engine,
                task_id=task_id,
                run_id=run_id,
                level="ERROR",
                event_type="grid_task_failed",
                message=str(e),
                payload={"traceback": traceback.format_exc()},
            )
            failed += 1
            if stop_on_error:
                break

    return {
        "grid_id": grid_id,
        "executed": success + failed,
        "success": success,
        "failed": failed,
        "pending_after_run": len(list_grid_tasks(engine, grid_id, status="pending", limit=100000)),
    }
