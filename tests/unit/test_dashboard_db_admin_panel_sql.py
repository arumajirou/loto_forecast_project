from __future__ import annotations

from loto_forecast.api.streamlit import dashboard_db_admin_panel as panel
from loto_forecast.api.streamlit import dashboard_db_admin_panel_sql as sql_mod


def test_database_schema_table_sql_builders() -> None:
    assert sql_mod.build_create_database_sql("sample", "owner", safe_ident=panel._safe_ident) == 'CREATE DATABASE "sample" OWNER "owner"'
    assert sql_mod.build_rename_database_sql("old", "new", safe_ident=panel._safe_ident) == 'ALTER DATABASE "old" RENAME TO "new"'
    assert sql_mod.build_drop_database_sql("sample", force=True, safe_ident=panel._safe_ident).endswith("WITH (FORCE)")
    assert sql_mod.build_create_schema_sql("public", safe_ident=panel._safe_ident) == 'CREATE SCHEMA IF NOT EXISTS "public"'
    assert sql_mod.build_drop_schema_sql("public", cascade=True, safe_ident=panel._safe_ident).endswith("CASCADE")
    assert sql_mod.build_rename_column_sql("public", "t", "a", "b", safe_ident=panel._safe_ident).endswith('RENAME COLUMN "a" TO "b"')
    assert sql_mod.build_drop_column_sql("public", "t", "a", cascade=True, safe_ident=panel._safe_ident).endswith("CASCADE")
    assert sql_mod.build_rename_table_sql("public", "t", "u", safe_ident=panel._safe_ident).endswith('RENAME TO "u"')
    assert sql_mod.build_drop_table_sql("public", "t", cascade=False, safe_ident=panel._safe_ident) == 'DROP TABLE IF EXISTS "public"."t"'


def test_build_run_sql_payload() -> None:
    assert sql_mod.build_run_sql_payload(" select 1 ") == {"raw": "select 1", "is_select": True}
    assert sql_mod.build_run_sql_payload("update x set y = 1")["is_select"] is False
