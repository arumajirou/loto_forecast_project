from __future__ import annotations

import pandas as pd
from _streamlit_test_double import FakeStreamlit

from loto_forecast.api.streamlit import dashboard_db_admin_panel as panel


def test_db_admin_renderer_renders_static_paths(monkeypatch) -> None:
    fake_st = FakeStreamlit(
        button_values={"Read Rows": True},
        text_values={"dbadmin_row_read_where_json": "{}"},
    )
    monkeypatch.setattr(panel, "st", fake_st)

    def query_df(_engine, sql: str, params=None):
        if "current_database()" in sql:
            return pd.DataFrame([{"database_name": "main", "current_user": "user", "server_time": "now"}])
        if "FROM pg_database" in sql:
            return pd.DataFrame([{"database_name": "main", "owner_name": "user"}])
        if "FROM information_schema.schemata" in sql:
            return pd.DataFrame([{"schema_name": "public", "owner_name": "user", "table_count": 1}])
        if "FROM information_schema.tables" in sql and "JOIN pg_class" in sql:
            return pd.DataFrame([{"table_schema": "public", "table_name": "items", "table_type": "BASE TABLE"}])
        if "SELECT * FROM" in sql:
            return pd.DataFrame([{"id": 1, "name": "alice"}])
        if "FROM information_schema.table_constraints" in sql and "FOREIGN KEY" not in sql:
            return pd.DataFrame([{"constraint_name": "pk_items", "constraint_type": "PRIMARY KEY"}])
        if "FOREIGN KEY" in sql:
            return pd.DataFrame(
                [{"src_schema": "public", "src_table": "items", "src_column": "id", "ref_schema": "public", "ref_table": "items_ref", "ref_column": "item_id", "constraint_name": "fk_items"}]
            )
        return pd.DataFrame()

    shown: list[pd.DataFrame] = []
    panel.render_db_admin_panel(
        engine=object(),
        database="main",
        row_limit=20,
        sample_limit=5,
        show_df=lambda df, **_: shown.append(df.copy()),
        query_df=query_df,
        table_columns=lambda *_args, **_kwargs: pd.DataFrame([{"column_name": "id"}, {"column_name": "name"}]),
        sample_table=lambda *_args, **_kwargs: pd.DataFrame([{"id": 1, "name": "alice"}]),
        exact_count=lambda *_args, **_kwargs: 1,
        clear_query_cache=lambda: None,
    )
    assert shown
    assert any(kind == "graphviz_chart" for kind, _ in fake_st.captured)
