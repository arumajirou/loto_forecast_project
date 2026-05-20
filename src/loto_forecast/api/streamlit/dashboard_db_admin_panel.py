from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import Engine

from loto_forecast.api.streamlit import dashboard_db_admin_panel_formatter as panel_formatter
from loto_forecast.api.streamlit import dashboard_db_admin_panel_helpers as helpers
from loto_forecast.api.streamlit import dashboard_db_admin_panel_sql as panel_sql
from loto_forecast.api.streamlit import dashboard_db_admin_panel_validator as panel_validator

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALLOWED_TYPES = [
    "BIGINT",
    "INTEGER",
    "SMALLINT",
    "TEXT",
    "BOOLEAN",
    "DATE",
    "TIMESTAMP",
    "TIMESTAMPTZ",
    "DOUBLE PRECISION",
    "NUMERIC",
    "JSONB",
    "UUID",
]


def _safe_ident(name: str) -> str:
    if not _IDENT_RE.fullmatch(str(name or "")):
        raise ValueError(f"unsafe identifier: {name}")
    return f'"{name}"'


def _normalize_type(type_name: str) -> str:
    t = str(type_name or "").strip().upper()
    if t not in _ALLOWED_TYPES:
        raise ValueError(f"unsupported type: {type_name}")
    return t


def _execute_sql(engine: Engine, sql: str, params: dict[str, Any] | None = None, autocommit: bool = False) -> int:
    if autocommit:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            res = conn.execute(text(sql), dict(params or {}))
            return int(res.rowcount or 0)
    with engine.begin() as conn:
        res = conn.execute(text(sql), dict(params or {}))
        return int(res.rowcount or 0)


def _list_schemas(engine: Engine, query_df: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return query_df(
        engine,
        """
        SELECT
          n.nspname AS schema_name,
          pg_get_userbyid(n.nspowner) AS owner,
          COUNT(c.oid) FILTER (WHERE c.relkind='r')::bigint AS table_count
        FROM pg_namespace n
        LEFT JOIN pg_class c ON c.relnamespace = n.oid
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
          AND n.nspname NOT LIKE 'pg_temp_%'
          AND n.nspname NOT LIKE 'pg_toast_temp_%'
        GROUP BY n.nspname, n.nspowner
        ORDER BY n.nspname
        """,
    )


def _list_tables(engine: Engine, query_df: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return query_df(
        engine,
        """
        SELECT
          t.table_schema,
          t.table_name,
          COALESCE(s.n_live_tup::bigint, c.reltuples::bigint, 0) AS est_rows,
          pg_total_relation_size(c.oid)::bigint AS total_bytes
        FROM information_schema.tables t
        JOIN pg_class c ON c.relname = t.table_name
        JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
        WHERE t.table_type='BASE TABLE'
          AND t.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY t.table_schema, t.table_name
        """,
    )


def _list_databases(engine: Engine, query_df: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return query_df(
        engine,
        """
        SELECT
          datname AS database_name,
          pg_get_userbyid(datdba) AS owner,
          datallowconn,
          datistemplate
        FROM pg_database
        ORDER BY datname
        """,
    )


def _schema_table_map(table_df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if table_df.empty:
        return out
    for schema, g in table_df.groupby("table_schema", sort=True):
        out[str(schema)] = sorted(g["table_name"].astype(str).tolist())
    return out


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    text_raw = str(raw or "").strip()
    if not text_raw:
        return {}
    try:
        obj = json.loads(text_raw)
    except Exception as e:
        raise ValueError(f"{label} json parse error: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be json object")
    return dict(obj)


def _validate_payload_columns(payload: dict[str, Any], allowed_columns: set[str], label: str) -> dict[str, Any]:
    cleaned = {str(k): v for k, v in payload.items()}
    invalid_cols = sorted([c for c in cleaned if c not in allowed_columns])
    if invalid_cols:
        raise ValueError(f"{label} contains unknown columns: {', '.join(invalid_cols)}")
    return cleaned


def _build_where_equals_clause(where_payload: dict[str, Any], param_prefix: str = "w") -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for idx, (col, value) in enumerate(where_payload.items()):
        col_ident = _safe_ident(str(col))
        if value is None:
            clauses.append(f"{col_ident} IS NULL")
            continue
        pkey = f"{param_prefix}_{idx}"
        clauses.append(f"{col_ident} = :{pkey}")
        params[pkey] = value
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def _build_er_dot(
    table_df: pd.DataFrame, fk_df: pd.DataFrame, selected_schemas: list[str], max_tables: int = 120
) -> str:
    def _esc(v: str) -> str:
        return str(v).replace("\\", "\\\\").replace('"', r"\"")

    use_tables = table_df[table_df["table_schema"].astype(str).isin(selected_schemas)].copy()
    use_tables = use_tables.head(int(max_tables))
    nodes = [f"{str(r.table_schema)}.{str(r.table_name)}" for r in use_tables.itertuples(index=False)]
    node_set = set(nodes)

    lines = [
        "digraph ER {",
        "  rankdir=LR;",
        "  graph [fontname=Helvetica, fontsize=11];",
        "  node [shape=box, style=rounded, fontname=Helvetica, fontsize=10];",
        "  edge [fontname=Helvetica, fontsize=9];",
    ]

    for node in nodes:
        lines.append(f'  "{_esc(node)}";')

    if not fk_df.empty:
        for r in fk_df.itertuples(index=False):
            src = f"{str(r.src_schema)}.{str(r.src_table)}"
            dst = f"{str(r.ref_schema)}.{str(r.ref_table)}"
            if src not in node_set or dst not in node_set:
                continue
            label = f"{str(r.src_column)}->{str(r.ref_column)}"
            lines.append(f'  "{_esc(src)}" -> "{_esc(dst)}" [label="{_esc(label)}"];')

    lines.append("}")
    return "\n".join(lines)


def render_db_admin_panel(
    *,
    engine: Engine,
    database: str,
    row_limit: int,
    sample_limit: int,
    show_df: Callable[..., None],
    query_df: Callable[..., pd.DataFrame],
    table_columns: Callable[..., pd.DataFrame],
    sample_table: Callable[..., pd.DataFrame],
    exact_count: Callable[..., int],
    clear_query_cache: Callable[[], None],
) -> None:
    st.subheader("DB管理 / CRAD / ER")
    st.caption("DB・スキーマ・テーブルのCRAD操作、テーブル確認、ER図表示を行います。")

    tab_db, tab_schema, tab_table, tab_row, tab_inspect, tab_er, tab_sql = st.tabs(
        ["DB", "Schema", "Table", "テーブル行CRAD", "テーブル確認", "ER図", "SQL"]
    )

    with tab_db:
        c1, c2 = st.columns(2)
        status_df = query_df(
            engine, "SELECT current_database() AS database_name, current_user AS current_user, now() AS server_time"
        )
        c1.metric("current database", str(status_df.iloc[0]["database_name"]) if not status_df.empty else str(database))
        c2.metric("current user", str(status_df.iloc[0]["current_user"]) if not status_df.empty else "-")
        db_df = _list_databases(engine, query_df=query_df)
        show_df(db_df, hide_index=True)

        with st.expander("Create Database", expanded=False):
            create_db_name = st.text_input("database name", value="", key="dbadmin_create_db_name")
            create_db_owner = st.text_input("owner (optional)", value="", key="dbadmin_create_db_owner")
            if st.button("Create DB", key="dbadmin_create_db_btn"):
                try:
                    sql = panel_sql.build_create_database_sql(create_db_name, create_db_owner, safe_ident=_safe_ident)
                    _execute_sql(engine, sql, autocommit=True)
                    clear_query_cache()
                    st.success(f"created: {create_db_name}")
                except Exception as e:
                    st.error(str(e))

        with st.expander("Alter Database (Rename)", expanded=False):
            old_db = st.text_input("old database", value="", key="dbadmin_rename_old_db")
            new_db = st.text_input("new database", value="", key="dbadmin_rename_new_db")
            if st.button("Rename DB", key="dbadmin_rename_db_btn"):
                try:
                    old_name = panel_validator.validate_non_empty_name(old_db, label="old database")
                    new_name = panel_validator.validate_non_empty_name(new_db, label="new database")
                    panel_validator.ensure_not_current_database(old_name, database, action="rename")
                    _execute_sql(engine, panel_sql.build_rename_database_sql(old_name, new_name, safe_ident=_safe_ident), autocommit=True)
                    clear_query_cache()
                    st.success(f"renamed: {old_name} -> {new_name}")
                except Exception as e:
                    st.error(str(e))

        with st.expander("Delete Database", expanded=False):
            drop_db = st.text_input("drop database", value="", key="dbadmin_drop_db_name")
            drop_force = st.toggle("force", value=False, key="dbadmin_drop_db_force")
            confirm_drop_db = st.text_input(
                "confirm text",
                value="",
                placeholder="DROP <database_name>",
                key="dbadmin_drop_db_confirm",
            )
            if st.button("Drop DB", key="dbadmin_drop_db_btn"):
                try:
                    drop_name = panel_validator.validate_non_empty_name(drop_db, label="drop database")
                    panel_validator.ensure_not_current_database(drop_name, database, action="drop")
                    helpers.validate_confirmation(panel_validator.expected_confirmation("DROP", drop_name), confirm_drop_db)
                    sql = panel_sql.build_drop_database_sql(drop_name, force=bool(drop_force), safe_ident=_safe_ident)
                    _execute_sql(engine, sql, autocommit=True)
                    clear_query_cache()
                    st.success(f"dropped: {drop_name}")
                except Exception as e:
                    st.error(str(e))

    with tab_schema:
        schema_df = _list_schemas(engine, query_df=query_df)
        show_df(schema_df, hide_index=True)

        with st.expander("Create Schema", expanded=False):
            create_schema_name = st.text_input("schema name", value="", key="dbadmin_create_schema_name")
            if st.button("Create Schema", key="dbadmin_create_schema_btn"):
                try:
                    schema_name = panel_validator.validate_non_empty_name(create_schema_name, label="schema name")
                    _execute_sql(engine, panel_sql.build_create_schema_sql(schema_name, safe_ident=_safe_ident))
                    clear_query_cache()
                    st.success(f"created schema: {create_schema_name}")
                except Exception as e:
                    st.error(str(e))

        with st.expander("Alter Schema (Rename)", expanded=False):
            old_schema = st.text_input("old schema", value="", key="dbadmin_rename_old_schema")
            new_schema = st.text_input("new schema", value="", key="dbadmin_rename_new_schema")
            if st.button("Rename Schema", key="dbadmin_rename_schema_btn"):
                try:
                    old_name = panel_validator.validate_non_empty_name(old_schema, label="old schema")
                    new_name = panel_validator.validate_non_empty_name(new_schema, label="new schema")
                    _execute_sql(engine, panel_sql.build_rename_schema_sql(old_name, new_name, safe_ident=_safe_ident))
                    clear_query_cache()
                    st.success(f"renamed schema: {old_schema} -> {new_schema}")
                except Exception as e:
                    st.error(str(e))

        with st.expander("Delete Schema", expanded=False):
            drop_schema_name = st.text_input("drop schema", value="", key="dbadmin_drop_schema_name")
            drop_schema_cascade = st.toggle("cascade", value=False, key="dbadmin_drop_schema_cascade")
            confirm_drop_schema = st.text_input(
                "confirm text",
                value="",
                placeholder="DROP <schema_name>",
                key="dbadmin_drop_schema_confirm",
            )
            if st.button("Drop Schema", key="dbadmin_drop_schema_btn"):
                try:
                    target_schema = panel_validator.validate_non_empty_name(drop_schema_name, label="drop schema")
                    helpers.validate_confirmation(
                        panel_validator.expected_confirmation("DROP", target_schema),
                        confirm_drop_schema,
                    )
                    sql = panel_sql.build_drop_schema_sql(
                        target_schema,
                        cascade=bool(drop_schema_cascade),
                        safe_ident=_safe_ident,
                    )
                    _execute_sql(engine, sql)
                    clear_query_cache()
                    st.success(f"dropped schema: {target_schema}")
                except Exception as e:
                    st.error(str(e))

    table_df = _list_tables(engine, query_df=query_df)
    table_map = _schema_table_map(table_df)
    _ = panel_formatter.build_table_options(table_map)
    schema_options = sorted(table_map.keys())

    with tab_table:
        st.markdown("**Table Create**")
        if not table_map:
            st.info("操作対象テーブルがありません。Schema作成後に再読み込みしてください。")
        create_cols_default = pd.DataFrame(
            [
                {"name": "id", "type": "BIGINT", "nullable": False, "primary_key": True, "unique": False},
                {"name": "created_at", "type": "TIMESTAMPTZ", "nullable": False, "primary_key": False, "unique": False},
            ]
        )
        create_schema = st.selectbox(
            "create schema",
            schema_options if schema_options else ["public"],
            index=0,
            key="dbadmin_create_table_schema",
        )
        create_table_name = st.text_input("table name", value="", key="dbadmin_create_table_name")
        create_cols = st.data_editor(
            create_cols_default,
            num_rows="dynamic",
            key="dbadmin_create_table_cols",
            column_config={
                "type": st.column_config.SelectboxColumn("type", options=_ALLOWED_TYPES),
                "nullable": st.column_config.CheckboxColumn("nullable"),
                "primary_key": st.column_config.CheckboxColumn("primary_key"),
                "unique": st.column_config.CheckboxColumn("unique"),
            },
            width="stretch",
            hide_index=True,
        )
        if st.button("Create Table", key="dbadmin_create_table_btn"):
            try:
                sql = helpers.build_create_table_sql(
                    create_schema,
                    create_table_name,
                    create_cols.to_dict(orient="records"),
                    safe_ident=_safe_ident,
                    normalize_type=_normalize_type,
                )
                _execute_sql(engine, sql)
                clear_query_cache()
                st.success(f"created table: {create_schema}.{create_table_name}")
            except Exception as e:
                st.error(str(e))

        st.markdown("**Table Alter**")
        if table_map:
            alt_schema = st.selectbox("alter schema", schema_options, index=0, key="dbadmin_alter_schema")
            alt_table = st.selectbox("alter table", table_map[alt_schema], index=0, key="dbadmin_alter_table")

            a1, a2 = st.columns(2)
            with a1:
                st.markdown("Add Column")
                add_col_name = st.text_input("column name", value="", key="dbadmin_add_col_name")
                add_col_type = st.selectbox("column type", _ALLOWED_TYPES, index=0, key="dbadmin_add_col_type")
                add_col_nullable = st.toggle("nullable", value=True, key="dbadmin_add_col_nullable")
                if st.button("Add Column", key="dbadmin_add_col_btn"):
                    try:
                        sql = helpers.build_add_column_sql(
                            alt_schema,
                            alt_table,
                            add_col_name,
                            add_col_type,
                            nullable=add_col_nullable,
                            safe_ident=_safe_ident,
                            normalize_type=_normalize_type,
                        )
                        _execute_sql(engine, sql)
                        clear_query_cache()
                        st.success(f"added column: {add_col_name}")
                    except Exception as e:
                        st.error(str(e))

            with a2:
                st.markdown("Rename Column")
                col_df = table_columns(engine, alt_schema, alt_table)
                col_opts = col_df["column_name"].astype(str).tolist() if not col_df.empty else []
                old_col = st.selectbox("old column", col_opts if col_opts else [""], index=0, key="dbadmin_old_col")
                new_col = st.text_input("new column", value="", key="dbadmin_new_col")
                if st.button("Rename Column", key="dbadmin_rename_col_btn"):
                    try:
                        sql = panel_sql.build_rename_column_sql(
                            alt_schema, alt_table, old_col, new_col, safe_ident=_safe_ident
                        )
                        _execute_sql(engine, sql)
                        clear_query_cache()
                        st.success(f"renamed column: {old_col} -> {new_col}")
                    except Exception as e:
                        st.error(str(e))

            b1, b2 = st.columns(2)
            with b1:
                st.markdown("Drop Column")
                col_df = table_columns(engine, alt_schema, alt_table)
                col_opts = col_df["column_name"].astype(str).tolist() if not col_df.empty else []
                drop_col = st.selectbox("drop column", col_opts if col_opts else [""], index=0, key="dbadmin_drop_col")
                drop_col_cascade = st.toggle("cascade", value=False, key="dbadmin_drop_col_cascade")
                if st.button("Drop Column", key="dbadmin_drop_col_btn"):
                    try:
                        sql = panel_sql.build_drop_column_sql(
                            alt_schema,
                            alt_table,
                            drop_col,
                            cascade=bool(drop_col_cascade),
                            safe_ident=_safe_ident,
                        )
                        _execute_sql(engine, sql)
                        clear_query_cache()
                        st.success(f"dropped column: {drop_col}")
                    except Exception as e:
                        st.error(str(e))

            with b2:
                st.markdown("Rename Table")
                new_table_name = st.text_input("new table name", value="", key="dbadmin_rename_table_name")
                if st.button("Rename Table", key="dbadmin_rename_table_btn"):
                    try:
                        sql = panel_sql.build_rename_table_sql(
                            alt_schema, alt_table, new_table_name, safe_ident=_safe_ident
                        )
                        _execute_sql(engine, sql)
                        clear_query_cache()
                        st.success(f"renamed table: {alt_table} -> {new_table_name}")
                    except Exception as e:
                        st.error(str(e))

            st.markdown("**Table Delete**")
            drop_table_cascade = st.toggle("drop table cascade", value=False, key="dbadmin_drop_table_cascade")
            confirm_drop_table = st.text_input(
                "confirm text",
                value="",
                placeholder=f"DROP {alt_schema}.{alt_table}",
                key="dbadmin_drop_table_confirm",
            )
            if st.button("Drop Table", key="dbadmin_drop_table_btn"):
                try:
                    expected = panel_validator.expected_confirmation("DROP", f"{alt_schema}.{alt_table}")
                    helpers.validate_confirmation(expected, confirm_drop_table)
                    sql = panel_sql.build_drop_table_sql(
                        alt_schema,
                        alt_table,
                        cascade=bool(drop_table_cascade),
                        safe_ident=_safe_ident,
                    )
                    _execute_sql(engine, sql)
                    clear_query_cache()
                    st.success(f"dropped table: {alt_schema}.{alt_table}")
                except Exception as e:
                    st.error(str(e))

    with tab_row:
        st.markdown("**テーブル行CRAD**")
        st.caption("JSON入力で行データの Create/Read/Update/Delete を実行します。")
        if not table_map:
            st.info("操作対象テーブルがありません。")
        else:
            row_schema = st.selectbox("schema", schema_options, index=0, key="dbadmin_row_schema")
            row_table = st.selectbox("table", table_map[row_schema], index=0, key="dbadmin_row_table")
            row_col_df = table_columns(engine, row_schema, row_table)
            row_columns = row_col_df["column_name"].astype(str).tolist() if not row_col_df.empty else []
            allowed_columns = set(row_columns)
            columns_caption = panel_formatter.build_columns_caption(row_columns)
            if columns_caption is not None:
                st.caption(columns_caption)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("Create (INSERT)")
                insert_json = st.text_area(
                    "insert values (json)",
                    value="{}",
                    height=120,
                    key="dbadmin_row_insert_json",
                )
                if st.button("Insert Row", key="dbadmin_row_insert_btn"):
                    try:
                        payload = _parse_json_object(insert_json, "insert values")
                        if not payload:
                            raise ValueError("insert values is empty")
                        payload = _validate_payload_columns(payload, allowed_columns, "insert values")
                        sql, params = helpers.build_insert_sql(
                            row_schema,
                            row_table,
                            payload,
                            safe_ident=_safe_ident,
                        )
                        n = _execute_sql(
                            engine,
                            sql,
                            params=params,
                        )
                        clear_query_cache()
                        st.success(f"inserted rows: {n}")
                    except Exception as e:
                        st.error(str(e))

            with r2:
                st.markdown("Read (SELECT)")
                read_where_json = st.text_area(
                    "where equals (json)",
                    value="{}",
                    height=120,
                    key="dbadmin_row_read_where_json",
                )
                read_limit = st.number_input(
                    "limit",
                    min_value=1,
                    max_value=max(1000, int(row_limit) * 10),
                    value=max(20, int(sample_limit)),
                    step=10,
                    key="dbadmin_row_read_limit",
                )
                if st.button("Read Rows", key="dbadmin_row_read_btn"):
                    try:
                        where_payload = _parse_json_object(read_where_json, "read where")
                        where_payload = _validate_payload_columns(where_payload, allowed_columns, "read where")
                        sql, where_params = helpers.build_select_sql(
                            row_schema,
                            row_table,
                            where_payload,
                            limit=int(read_limit),
                            safe_ident=_safe_ident,
                            build_where_equals_clause=_build_where_equals_clause,
                        )
                        out_df = query_df(engine, sql, where_params)
                        show_df(out_df, hide_index=True)
                    except Exception as e:
                        st.error(str(e))

            u1, u2 = st.columns(2)
            with u1:
                st.markdown("Update (UPDATE)")
                update_set_json = st.text_area(
                    "set values (json)",
                    value="{}",
                    height=120,
                    key="dbadmin_row_update_set_json",
                )
                update_where_json = st.text_area(
                    "where equals (json)",
                    value="{}",
                    height=120,
                    key="dbadmin_row_update_where_json",
                )
                update_all = st.toggle("allow update all rows", value=False, key="dbadmin_row_update_all")
                if st.button("Update Rows", key="dbadmin_row_update_btn"):
                    try:
                        set_payload = _parse_json_object(update_set_json, "update set")
                        if not set_payload:
                            raise ValueError("update set is empty")
                        where_payload = _parse_json_object(update_where_json, "update where")
                        set_payload = _validate_payload_columns(set_payload, allowed_columns, "update set")
                        where_payload = _validate_payload_columns(where_payload, allowed_columns, "update where")
                        panel_validator.validate_bulk_change_allowed(
                            where_payload,
                            allow_all=bool(update_all),
                            action="update",
                        )
                        sql, params = helpers.build_update_sql(
                            row_schema,
                            row_table,
                            set_payload,
                            where_payload,
                            safe_ident=_safe_ident,
                            build_where_equals_clause=_build_where_equals_clause,
                        )
                        n = _execute_sql(engine, sql, params=params)
                        clear_query_cache()
                        st.success(f"updated rows: {n}")
                    except Exception as e:
                        st.error(str(e))

            with u2:
                st.markdown("Delete (DELETE)")
                delete_where_json = st.text_area(
                    "where equals (json)",
                    value="{}",
                    height=120,
                    key="dbadmin_row_delete_where_json",
                )
                delete_all = st.toggle("allow delete all rows", value=False, key="dbadmin_row_delete_all")
                delete_confirm = st.text_input(
                    "confirm text",
                    value="",
                    placeholder=f"DELETE {row_schema}.{row_table}",
                    key="dbadmin_row_delete_confirm",
                )
                if st.button("Delete Rows", key="dbadmin_row_delete_btn"):
                    try:
                        where_payload = _parse_json_object(delete_where_json, "delete where")
                        where_payload = _validate_payload_columns(where_payload, allowed_columns, "delete where")
                        panel_validator.validate_bulk_change_allowed(
                            where_payload,
                            allow_all=bool(delete_all),
                            action="delete",
                        )
                        expected = panel_validator.expected_confirmation("DELETE", f"{row_schema}.{row_table}")
                        helpers.validate_confirmation(expected, delete_confirm)
                        sql, where_params = helpers.build_delete_sql(
                            row_schema,
                            row_table,
                            where_payload,
                            safe_ident=_safe_ident,
                            build_where_equals_clause=_build_where_equals_clause,
                        )
                        n = _execute_sql(
                            engine,
                            sql,
                            params=where_params,
                        )
                        clear_query_cache()
                        st.success(f"deleted rows: {n}")
                    except Exception as e:
                        st.error(str(e))

    with tab_inspect:
        st.markdown("**テーブル確認**")
        if not table_map:
            st.info("確認対象テーブルがありません。")
        else:
            schema = st.selectbox("schema", schema_options, index=0, key="dbadmin_inspect_schema")
            table = st.selectbox("table", table_map[schema], index=0, key="dbadmin_inspect_table")
            show_df(table_columns(engine, schema, table), hide_index=True)
            if st.button("Count rows (exact)", key="dbadmin_inspect_count_btn"):
                try:
                    st.metric("row_count", exact_count(engine, schema, table))
                except Exception as e:
                    st.error(str(e))
            show_df(sample_table(engine, schema, table, sample_limit), hide_index=True)

            cons_df = query_df(
                engine,
                """
                SELECT
                  tc.constraint_name,
                  tc.constraint_type,
                  string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns,
                  ccu.table_schema AS ref_schema,
                  ccu.table_name AS ref_table,
                  ccu.column_name AS ref_column
                FROM information_schema.table_constraints tc
                LEFT JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                LEFT JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                 AND tc.table_schema = ccu.table_schema
                WHERE tc.table_schema = :schema AND tc.table_name = :table
                GROUP BY tc.constraint_name, tc.constraint_type, ccu.table_schema, ccu.table_name, ccu.column_name
                ORDER BY tc.constraint_type, tc.constraint_name
                """,
                {"schema": schema, "table": table},
            )
            st.markdown("**Constraints**")
            show_df(cons_df, hide_index=True)

    with tab_er:
        st.markdown("**ER図**")
        if table_df.empty:
            st.info("ER図対象のテーブルがありません。")
        else:
            schema_opts = sorted(table_df["table_schema"].astype(str).unique().tolist())
            default_schemas = panel_formatter.default_er_schemas(table_df)
            selected_schemas = st.multiselect("schemas", schema_opts, default=default_schemas, key="dbadmin_er_schemas")
            max_tables = st.slider(
                "max tables",
                min_value=10,
                max_value=300,
                value=min(120, max(20, row_limit * 2)),
                step=10,
                key="dbadmin_er_max_tables",
            )

            fk_df = query_df(
                engine,
                """
                SELECT
                  tc.table_schema AS src_schema,
                  tc.table_name AS src_table,
                  kcu.column_name AS src_column,
                  ccu.table_schema AS ref_schema,
                  ccu.table_name AS ref_table,
                  ccu.column_name AS ref_column,
                  tc.constraint_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                 AND tc.table_name = kcu.table_name
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                 AND tc.table_schema = ccu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                ORDER BY tc.table_schema, tc.table_name, tc.constraint_name
                """,
            )

            if not selected_schemas:
                st.info("schema を1つ以上選択してください。")
            else:
                dot = _build_er_dot(table_df, fk_df, selected_schemas, max_tables=max_tables)
                try:
                    st.graphviz_chart(dot, width="stretch")
                except Exception as e:
                    st.warning(f"graphviz描画に失敗したためDOT文字列を表示します: {e}")
                    st.code(dot, language="dot")

                fk_show = panel_formatter.filter_fk_rows(fk_df, selected_schemas=selected_schemas)
                st.markdown("**Foreign Key一覧**")
                show_df(fk_show.head(max(1000, int(row_limit * 10))), hide_index=True)

    with tab_sql:
        st.markdown("**管理SQL実行**")
        st.caption("SELECT/DDL/DML を実行できます。書き込み系は十分注意してください。")
        default_sql = "SELECT current_database(), current_user, now()"
        sql_text = st.text_area("sql", value=default_sql, height=140, key="dbadmin_sql_text")
        allow_write = st.toggle("allow write (INSERT/UPDATE/DELETE/DDL)", value=False, key="dbadmin_allow_write")
        if st.button("Run SQL", key="dbadmin_run_sql_btn"):
            sql_payload = panel_sql.build_run_sql_payload(sql_text)
            raw = sql_payload["raw"]
            if not raw:
                st.error("sql is empty")
            else:
                try:
                    if sql_payload["is_select"]:
                        out_df = query_df(engine, raw)
                        show_df(out_df, hide_index=True)
                    else:
                        if not allow_write:
                            st.error("write SQL is blocked. enable 'allow write' first.")
                        else:
                            n = _execute_sql(engine, raw)
                            clear_query_cache()
                            st.success(f"executed. affected rows: {n}")
                except Exception as e:
                    st.error(str(e))
