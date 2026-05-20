from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from loto_forecast.orchestration import grid_runner


def test_normalize_param_space_and_resource_summary() -> None:
    assert grid_runner._normalize_param_space({"a": 1, "b": [2, 3]}) == {"a": [1], "b": [2, 3]}
    assert grid_runner._resource_summary([]) == {}

    summary = grid_runner._resource_summary(
        [
            {"cpu_percent": 10.0, "mem_percent": 20.0, "rss_mb": 30.0},
            {"cpu_percent": 30.0, "mem_percent": 40.0, "rss_mb": 50.0},
        ]
    )
    assert summary["sample_count"] == 2
    assert summary["cpu_max"] == 30.0
    assert summary["rss_mean_mb"] == 40.0


def test_create_grid_writes_definition_and_tasks(monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(grid_runner, "make_engine", lambda: object())
    monkeypatch.setattr(grid_runner, "create_grid_definition", lambda *args, **kwargs: calls.setdefault("def", kwargs))
    monkeypatch.setattr(
        grid_runner, "replace_grid_tasks", lambda _engine, grid_id, tasks: calls.setdefault("tasks", (grid_id, tasks))
    )

    out = grid_runner.create_grid(
        grid_id="g1",
        library_name="lib",
        adapter_name="adapter",
        model_name="Model",
        horizon=3,
        param_space={"lr": [0.1, 0.2], "seed": 1},
        max_tasks=1,
    )

    assert out == {"grid_id": "g1", "task_count": 1, "model_name": "Model", "adapter_name": "adapter"}
    assert calls["def"]["grid_id"] == "g1"
    assert calls["tasks"][0] == "g1"
    assert calls["tasks"][1] == [{"lr": 0.1, "seed": 1}]


def test_run_grid_returns_when_grid_missing_or_no_pending(monkeypatch) -> None:
    monkeypatch.setattr(grid_runner, "make_engine", lambda: object())
    monkeypatch.setattr(grid_runner, "get_grid_definition", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="grid not found"):
        grid_runner.run_grid("missing")

    monkeypatch.setattr(
        grid_runner,
        "get_grid_definition",
        lambda *_args, **_kwargs: {
            "adapter_name": "adapter",
            "model_name": "Model",
            "horizon": 3,
            "run_predict": True,
            "run_evaluate": True,
            "library_name": "lib",
        },
    )
    monkeypatch.setattr("loto_forecast.models.registry.get_adapter", lambda _name: object())
    monkeypatch.setattr(grid_runner, "list_grid_tasks", lambda *_args, **_kwargs: [])

    out = grid_runner.run_grid("g1")
    assert out == {"grid_id": "g1", "message": "no pending tasks", "executed": 0}


@dataclass
class _Sample:
    cpu_percent: float
    mem_percent: float
    rss_mb: float


def test_run_grid_handles_success_and_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(grid_runner, "make_engine", lambda: object())
    monkeypatch.setattr(
        grid_runner,
        "get_grid_definition",
        lambda *_args, **_kwargs: {
            "adapter_name": "adapter",
            "model_name": "Model",
            "horizon": 2,
            "run_predict": True,
            "run_evaluate": True,
            "library_name": "lib",
        },
    )
    pending = [
        {"task_id": 1, "task_order": 1, "param_values": {"ok": True}},
        {"task_id": 2, "task_order": 2, "param_values": '{"ok": false}'},
    ]
    pending_after = [[]]

    def _fake_list_tasks(_engine, _grid_id, status="pending", limit=100000):
        if status == "pending" and pending_after:
            return pending if limit == 100000 and len(pending_after) == 1 else pending_after.pop(0)
        return []

    monkeypatch.setattr(grid_runner, "list_grid_tasks", _fake_list_tasks)
    monkeypatch.setattr(grid_runner, "setup_logging", lambda run_id: tmp_path / f"{run_id}.log")
    monkeypatch.setattr(grid_runner, "sample_resources", lambda: _Sample(10.0, 20.0, 30.0))

    calls = {"start": [], "finish": [], "events": [], "mark_end": [], "write_samples": [], "upsert": []}
    monkeypatch.setattr(
        grid_runner, "start_grid_task", lambda _engine, task_id, run_id, log_path: calls["start"].append((task_id, run_id))
    )
    monkeypatch.setattr(grid_runner, "upsert_model_run", lambda *args, **kwargs: calls["upsert"].append(kwargs))
    monkeypatch.setattr(
        grid_runner,
        "finish_grid_task",
        lambda _engine, task_id, status, result, metrics, resource_summary, error_message: calls["finish"].append(
            (task_id, status, result, metrics, error_message)
        ),
    )
    monkeypatch.setattr(
        grid_runner,
        "log_execution_event",
        lambda _engine, task_id, run_id, event_type, message, payload=None, level="INFO": calls["events"].append(
            (task_id, event_type, level)
        ),
    )
    monkeypatch.setattr(
        grid_runner, "mark_model_run_end", lambda _engine, run_id, status, error_message: calls["mark_end"].append(status)
    )
    monkeypatch.setattr(
        grid_runner, "write_resource_samples", lambda _engine, run_id, samples: calls["write_samples"].append((run_id, len(samples)))
    )

    class _Adapter:
        def validate(self, model_name, model_params):
            return {"ok": bool(model_params.get("ok")), "errors": [] if model_params.get("ok") else ["bad"]}

        def run(self, **kwargs):
            return SimpleNamespace(
                run_id=kwargs["run_id"],
                train={"rows": 10},
                predict={"rows": 2},
                evaluate={"metrics": {"mae": 1.0}},
            )

    monkeypatch.setattr("loto_forecast.models.registry.get_adapter", lambda _name: _Adapter())

    out = grid_runner.run_grid("g1")

    assert out["executed"] == 2
    assert out["success"] == 1
    assert out["failed"] == 1
    assert calls["finish"][0][1] == "success"
    assert calls["finish"][1][1] == "failed"
    assert "failed" in calls["mark_end"]


def test_run_grid_stop_on_error_breaks_after_first_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(grid_runner, "make_engine", lambda: object())
    monkeypatch.setattr(
        grid_runner,
        "get_grid_definition",
        lambda *_args, **_kwargs: {
            "adapter_name": "adapter",
            "model_name": "Model",
            "horizon": 2,
            "run_predict": True,
            "run_evaluate": True,
            "library_name": "lib",
        },
    )
    monkeypatch.setattr(
        grid_runner,
        "list_grid_tasks",
        lambda *_args, **_kwargs: [{"task_id": 1, "task_order": 1, "param_values": {"ok": False}}],
    )
    monkeypatch.setattr(grid_runner, "setup_logging", lambda run_id: tmp_path / f"{run_id}.log")
    monkeypatch.setattr(grid_runner, "sample_resources", lambda: _Sample(10.0, 20.0, 30.0))
    monkeypatch.setattr(grid_runner, "start_grid_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grid_runner, "upsert_model_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grid_runner, "write_resource_samples", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grid_runner, "mark_model_run_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grid_runner, "finish_grid_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grid_runner, "log_execution_event", lambda *_args, **_kwargs: None)

    class _Adapter:
        def validate(self, model_name, model_params):
            return {"ok": False, "errors": ["bad"]}

    monkeypatch.setattr("loto_forecast.models.registry.get_adapter", lambda _name: _Adapter())

    out = grid_runner.run_grid("g1", stop_on_error=True)

    assert out["executed"] == 1
    assert out["failed"] == 1
