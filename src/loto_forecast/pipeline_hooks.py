from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def demo_train_and_predict(params: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(params.get("seed", 42)))
    n = int(params.get("n", 240))
    x = np.linspace(0, 12, n)
    y_true = np.sin(x) + 0.1 * rng.normal(size=n)
    y_pred = np.sin(x + 0.2)
    exog = pd.DataFrame(
        {
            "x1": np.cos(x),
            "x2": np.sin(x * 0.5),
            "x3": rng.normal(size=n),
        }
    )
    metrics = {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }
    return {
        "model": {
            "name": str(params.get("model_name", "demo_model")),
            "version": "0.1",
            "family": "baseline",
            "properties": {"note": "replace demo_train_and_predict with your real training pipeline"},
            "hyperparams": dict(params.get("hyperparams", {})),
        },
        "evaluation": {
            "dataset_id": str(params.get("dataset_id", "demo_dataset")),
            "metrics": metrics,
            "notes": "demo run",
        },
        "pred": {
            "t": [str(i) for i in range(n)],
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        },
        "X_exog": exog.to_dict(orient="list"),
        "y_exog_target": y_true.tolist(),
    }
