from pathlib import Path

import pytest

from loto_forecast.cli import _load_json_any, _load_json_arg


def test_load_json_arg_default_copy():
    out = _load_json_arg(None, default={"a": 1})
    assert out == {"a": 1}
    out["a"] = 2
    assert _load_json_arg(None, default={"a": 1}) == {"a": 1}


def test_load_json_arg_from_json_string():
    assert _load_json_arg('{"backend":"optuna","num_samples":20}') == {
        "backend": "optuna",
        "num_samples": 20,
    }


def test_load_json_arg_from_python_literal_string():
    assert _load_json_arg("{'backend':'optuna','num_samples':20}") == {
        "backend": "optuna",
        "num_samples": 20,
    }


def test_load_json_arg_from_file(tmp_path: Path):
    p = tmp_path / "params.json"
    p.write_text('{"x": 1}', encoding="utf-8")
    assert _load_json_arg(str(p)) == {"x": 1}


def test_load_json_arg_rejects_non_dict():
    with pytest.raises(ValueError):
        _load_json_arg("[1,2,3]")


def test_load_json_any_supports_list():
    assert _load_json_any("[1,2,3]", default=[]) == [1, 2, 3]
