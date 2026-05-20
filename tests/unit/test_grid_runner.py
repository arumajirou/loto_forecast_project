from loto_forecast.orchestration.grid_runner import expand_param_grid


def test_expand_param_grid_product():
    param_space = {
        "num_samples": [5, 10],
        "seed": [1, 2],
        "backend": ["optuna"],
    }
    out = expand_param_grid(param_space)
    assert len(out) == 4
    assert {tuple(sorted(x.items())) for x in out}


def test_expand_param_grid_max_tasks():
    param_space = {"a": [1, 2, 3], "b": [10, 20]}
    out = expand_param_grid(param_space, max_tasks=2)
    assert len(out) == 2
