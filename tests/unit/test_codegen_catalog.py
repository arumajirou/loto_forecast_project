from pathlib import Path

from loto_forecast.catalog.codegen_catalog import load_codegen_yaml, parse_codegen_rows


def test_parse_codegen_rows_basic(tmp_path: Path):
    yaml_path = tmp_path / "sample_codegen.yaml"
    yaml_path.write_text(
        """
title: sample
bundle_kind: code
count: 2
rows:
  -
    type: function
    name: f1
    path: lib.mod.f1
    module: lib.mod
    library: lib
    params:
      - name: x
        kind: POSITIONAL_OR_KEYWORD
        annotation: int
        has_default: false
        default_repr: ""
  -
    type: method
    name: fit
    path: lib.mod.ClassA.fit
    module: lib.mod
    library: lib
    params: []
""",
        encoding="utf-8",
    )

    payload = load_codegen_yaml(yaml_path)
    rows = parse_codegen_rows(payload)

    assert len(rows) == 2
    assert rows[0].symbol_name == "f1"
    assert rows[0].params[0].param_name == "x"
    assert rows[0].params[0].is_required is True

    assert rows[1].symbol_type == "method"
    assert rows[1].parent_symbol == "lib.mod.ClassA"
