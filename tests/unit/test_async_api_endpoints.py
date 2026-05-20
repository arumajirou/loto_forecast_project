from __future__ import annotations

from loto_forecast.api import server
from loto_forecast.api.server import RecursiveSubmitReq
from loto_forecast.infra.db import get_session, init_db
from loto_forecast.infra.orm_models import Evaluation, Model


def test_contract_and_drift_endpoints() -> None:
    init_db()
    with get_session() as s:
        m = Model(name="m1", version="0", family="demo", properties={}, hyperparams={})
        s.add(m)
        s.flush()
        ev = Evaluation(
            model_id=int(m.id),
            dataset_id="ds1",
            metrics={"mae": 0.1},
            notes=None,
            artifacts={},
            analysis={
                "explainability_contract": {"prediction_interval": {"coverage_target": 0.9}},
                "drift": {"has_reference": False},
            },
        )
        s.add(ev)
        s.commit()
        evaluation_id = int(ev.id)

    c = server.get_evaluation_contract(evaluation_id)
    d = server.get_evaluation_drift(evaluation_id)
    assert c["evaluation_id"] == evaluation_id
    assert "prediction_interval" in c["contract"]
    assert d["drift"]["has_reference"] is False


def test_submit_recursive_endpoint(monkeypatch) -> None:
    def _fake_submit_recursive_tasks(**kwargs):
        return {
            "loop_id": "loop-test-1",
            "recursive_depth": kwargs.get("recursive_depth", 1),
            "strategy": kwargs.get("strategy", "seed_increment"),
            "task_ids": ["t1", "t2"],
        }

    monkeypatch.setattr(server, "submit_recursive_tasks", _fake_submit_recursive_tasks)
    req = RecursiveSubmitReq(
        kind="train",
        callable="loto_forecast.pipeline_hooks:demo_train_and_predict",
        params={},
        recursive_depth=2,
    )
    out = server.submit_recursive(req)
    assert out["loop_id"] == "loop-test-1"
    assert out["recursive_depth"] == 2
    assert len(out["task_ids"]) == 2
