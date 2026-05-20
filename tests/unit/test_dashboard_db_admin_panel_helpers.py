from __future__ import annotations

import pytest

from loto_forecast.api.streamlit import dashboard_db_admin_panel as panel
from loto_forecast.api.streamlit import dashboard_db_admin_panel_formatter as formatter
from loto_forecast.api.streamlit import dashboard_db_admin_panel_helpers as helpers
from loto_forecast.api.streamlit import dashboard_db_admin_panel_sql as sql_mod
from loto_forecast.api.streamlit import dashboard_db_admin_panel_validator as validator


def test_validate_confirmation_and_create_table_sql() -> None:
    helpers.validate_confirmation("DROP db", "DROP db")
    with pytest.raises(ValueError):
        helpers.validate_confirmation("DROP db", "DROP other")

    sql = helpers.build_create_table_sql(
        "public",
        "sample_table",
        [
            {"name": "id", "type": "BIGINT", "nullable": False, "primary_key": True, "unique": False},
            {"name": "name", "type": "TEXT", "nullable": False, "primary_key": False, "unique": True},
        ],
        safe_ident=panel._safe_ident,
        normalize_type=panel._normalize_type,
    )
    assert 'CREATE TABLE "public"."sample_table"' in sql
    assert 'PRIMARY KEY ("id")' in sql
    assert '"name" TEXT NOT NULL UNIQUE' in sql


def test_add_column_and_row_crud_sql_builders() -> None:
    add_sql = helpers.build_add_column_sql(
        "public",
        "sample",
        "score",
        "INTEGER",
        nullable=False,
        safe_ident=panel._safe_ident,
        normalize_type=panel._normalize_type,
    )
    assert add_sql.endswith('"score" INTEGER NOT NULL')

    insert_sql, insert_params = helpers.build_insert_sql(
        "public",
        "sample",
        {"id": 1, "name": "alice"},
        safe_ident=panel._safe_ident,
    )
    assert insert_sql == 'INSERT INTO "public"."sample" ("id", "name") VALUES (:v_0, :v_1)'
    assert insert_params == {"v_0": 1, "v_1": "alice"}

    select_sql, select_params = helpers.build_select_sql(
        "public",
        "sample",
        {"id": 1},
        limit=20,
        safe_ident=panel._safe_ident,
        build_where_equals_clause=panel._build_where_equals_clause,
    )
    assert 'SELECT * FROM "public"."sample" WHERE "id" = :rw_0 LIMIT 20' == select_sql
    assert select_params == {"rw_0": 1}

    update_sql, update_params = helpers.build_update_sql(
        "public",
        "sample",
        {"name": "bob"},
        {"id": 1},
        safe_ident=panel._safe_ident,
        build_where_equals_clause=panel._build_where_equals_clause,
    )
    assert 'UPDATE "public"."sample" SET "name" = :sv_0 WHERE "id" = :uw_0' == update_sql
    assert update_params == {"sv_0": "bob", "uw_0": 1}

    delete_sql, delete_params = helpers.build_delete_sql(
        "public",
        "sample",
        {"id": 1},
        safe_ident=panel._safe_ident,
        build_where_equals_clause=panel._build_where_equals_clause,
    )
    assert 'DELETE FROM "public"."sample" WHERE "id" = :dw_0' == delete_sql
    assert delete_params == {"dw_0": 1}


def test_db_admin_panel_uses_helper_calls() -> None:
    source = open(panel.__file__, encoding="utf-8").read()
    assert "helpers.build_create_table_sql" in source
    assert "helpers.build_insert_sql" in source
    assert "helpers.build_update_sql" in source
    assert "helpers.build_delete_sql" in source
    assert "panel_sql.build_create_database_sql" in source
    assert "panel_formatter.default_er_schemas" in source
    assert "panel_validator.validate_bulk_change_allowed" in source


def test_db_admin_submodules_basic_paths() -> None:
    assert sql_mod.build_run_sql_payload("show tables")["is_select"] is True
    assert formatter.build_table_options({})["schemas"] == []
    assert validator.expected_confirmation("DELETE", "public.table") == "DELETE public.table"
