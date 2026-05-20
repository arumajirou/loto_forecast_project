from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from loto_forecast.infra.db import get_session, init_db
from loto_forecast.infra.orm_models import Evaluation, PredictionRow, ResourceSample, Task
from loto_forecast.services.task_runner import submit_recursive_tasks, submit_task

app = FastAPI(title="loto-forecast async backend")


class SubmitReq(BaseModel):
    kind: str = Field(default="train")
    callable: str = Field(..., description="package.module:function")
    params: dict[str, Any] = Field(default_factory=dict)
    num_cpus: float = 1.0
    num_gpus: float = 0.0


class RecursiveSubmitReq(BaseModel):
    kind: str = Field(default="train")
    callable: str = Field(..., description="package.module:function")
    params: dict[str, Any] = Field(default_factory=dict)
    recursive_depth: int = 3
    strategy: str = "seed_increment"
    seed_key: str = "seed"
    seed_start: int | None = None
    seed_step: int = 1
    num_cpus: float = 1.0
    num_gpus: float = 0.0


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/tasks/submit")
def submit(req: SubmitReq) -> dict[str, Any]:
    task_id = submit_task(
        kind=req.kind,
        callable_path=req.callable,
        params=req.params,
        num_cpus=float(req.num_cpus),
        num_gpus=float(req.num_gpus),
    )
    return {"task_id": task_id}


@app.post("/loops/submit")
def submit_recursive(req: RecursiveSubmitReq) -> dict[str, Any]:
    out = submit_recursive_tasks(
        kind=req.kind,
        callable_path=req.callable,
        params=req.params,
        recursive_depth=int(req.recursive_depth),
        num_cpus=float(req.num_cpus),
        num_gpus=float(req.num_gpus),
        strategy=str(req.strategy),
        seed_key=str(req.seed_key),
        seed_start=req.seed_start,
        seed_step=int(req.seed_step),
    )
    return out


@app.get("/tasks")
def list_tasks(limit: int = 50) -> dict[str, Any]:
    with get_session() as s:
        rows = s.query(Task).order_by(Task.created_at.desc()).limit(max(1, min(int(limit), 500))).all()
    return {
        "rows": [
            {
                "id": r.id,
                "kind": r.kind,
                "status": r.status,
                "created_at": r.created_at,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
            }
            for r in rows
        ]
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    with get_session() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "id": t.id,
            "kind": t.kind,
            "status": t.status,
            "created_at": t.created_at,
            "started_at": t.started_at,
            "finished_at": t.finished_at,
            "params": t.params,
            "result": t.result,
            "error": t.error,
        }


@app.get("/tasks/{task_id}/resources")
def get_task_resources(task_id: str, limit: int = 5000) -> dict[str, Any]:
    with get_session() as s:
        rows = (
            s.query(ResourceSample)
            .filter(ResourceSample.task_id == task_id)
            .order_by(ResourceSample.ts.asc())
            .limit(max(1, min(int(limit), 50000)))
            .all()
        )
    return {
        "rows": [
            {
                "ts": r.ts,
                "cpu_percent": r.cpu_percent,
                "rss_mb": r.rss_mb,
                "vms_mb": r.vms_mb,
                "gpu_util": r.gpu_util,
                "gpu_mem_mb": r.gpu_mem_mb,
                "gpu_temp_c": r.gpu_temp_c,
            }
            for r in rows
        ]
    }


@app.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: int) -> dict[str, Any]:
    with get_session() as s:
        ev = s.get(Evaluation, evaluation_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return {
            "id": ev.id,
            "model_id": ev.model_id,
            "dataset_id": ev.dataset_id,
            "metrics": ev.metrics,
            "notes": ev.notes,
            "artifacts": ev.artifacts,
            "analysis": ev.analysis,
            "created_at": ev.created_at,
        }


@app.get("/evaluations/{evaluation_id}/contract")
def get_evaluation_contract(evaluation_id: int) -> dict[str, Any]:
    with get_session() as s:
        ev = s.get(Evaluation, evaluation_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        analysis = dict(ev.analysis or {})
    return {
        "evaluation_id": int(evaluation_id),
        "contract": analysis.get("explainability_contract", {}),
    }


@app.get("/evaluations/{evaluation_id}/drift")
def get_evaluation_drift(evaluation_id: int) -> dict[str, Any]:
    with get_session() as s:
        ev = s.get(Evaluation, evaluation_id)
        if ev is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        analysis = dict(ev.analysis or {})
    return {
        "evaluation_id": int(evaluation_id),
        "drift": analysis.get("drift", {}),
    }


@app.get("/evaluations/{evaluation_id}/predictions")
def get_predictions(evaluation_id: int, limit: int = 100000) -> dict[str, Any]:
    with get_session() as s:
        rows = (
            s.query(PredictionRow)
            .filter(PredictionRow.evaluation_id == int(evaluation_id))
            .order_by(PredictionRow.id.asc())
            .limit(max(1, min(int(limit), 200000)))
            .all()
        )
    return {"rows": [{"id": r.id, "t": r.t, "y_true": r.y_true, "y_pred": r.y_pred} for r in rows]}
