from __future__ import annotations

import pandas as pd

from loto_forecast.api.streamlit import dashboard_db_admin_panel_formatter as formatter


def test_table_option_and_caption_formatters() -> None:
    options = formatter.build_table_options({"b": ["t2"], "a": ["t1", "t3"]})
    assert options["schemas"] == ["a", "b"]
    assert options["selected_schema"] == "a"
    assert options["selected_table"] == "t1"
    assert formatter.build_columns_caption(["id", "name"]) == "columns: id, name"
    assert formatter.build_columns_caption([]) is None


def test_er_schema_defaults_and_fk_filter() -> None:
    table_df = pd.DataFrame({"table_schema": ["resources", "meta", "other"]})
    assert formatter.default_er_schemas(table_df) == ["resources", "meta"]

    fk_df = pd.DataFrame(
        {
            "src_schema": ["resources", "other"],
            "ref_schema": ["meta", "other"],
            "src_table": ["a", "b"],
        }
    )
    filtered = formatter.filter_fk_rows(fk_df, selected_schemas=["resources", "meta"])
    assert filtered["src_schema"].tolist() == ["resources"]
