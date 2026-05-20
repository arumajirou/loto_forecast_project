from __future__ import annotations

from typing import Any


def build_create_database_sql(database_name: str, owner: str, *, safe_ident: Any) -> str:
    sql = f"CREATE DATABASE {safe_ident(database_name.strip())}"
    owner_raw = owner.strip()
    if owner_raw:
        sql += f" OWNER {safe_ident(owner_raw)}"
    return sql


def build_rename_database_sql(old_name: str, new_name: str, *, safe_ident: Any) -> str:
    return f"ALTER DATABASE {safe_ident(old_name.strip())} RENAME TO {safe_ident(new_name.strip())}"


def build_drop_database_sql(database_name: str, *, force: bool, safe_ident: Any) -> str:
    sql = f"DROP DATABASE IF EXISTS {safe_ident(database_name.strip())}"
    if force:
        sql += " WITH (FORCE)"
    return sql


def build_create_schema_sql(schema_name: str, *, safe_ident: Any) -> str:
    return f"CREATE SCHEMA IF NOT EXISTS {safe_ident(schema_name.strip())}"


def build_rename_schema_sql(old_name: str, new_name: str, *, safe_ident: Any) -> str:
    return f"ALTER SCHEMA {safe_ident(old_name.strip())} RENAME TO {safe_ident(new_name.strip())}"


def build_drop_schema_sql(schema_name: str, *, cascade: bool, safe_ident: Any) -> str:
    sql = f"DROP SCHEMA IF EXISTS {safe_ident(schema_name.strip())}"
    if cascade:
        sql += " CASCADE"
    return sql


def build_rename_column_sql(schema: str, table: str, old_col: str, new_col: str, *, safe_ident: Any) -> str:
    return (
        f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} "
        f"RENAME COLUMN {safe_ident(old_col)} TO {safe_ident(new_col.strip())}"
    )


def build_drop_column_sql(schema: str, table: str, column: str, *, cascade: bool, safe_ident: Any) -> str:
    sql = f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} DROP COLUMN IF EXISTS {safe_ident(column)}"
    if cascade:
        sql += " CASCADE"
    return sql


def build_rename_table_sql(schema: str, table: str, new_table: str, *, safe_ident: Any) -> str:
    return (
        f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} "
        f"RENAME TO {safe_ident(new_table.strip())}"
    )


def build_drop_table_sql(schema: str, table: str, *, cascade: bool, safe_ident: Any) -> str:
    sql = f"DROP TABLE IF EXISTS {safe_ident(schema)}.{safe_ident(table)}"
    if cascade:
        sql += " CASCADE"
    return sql


def build_run_sql_payload(raw_sql: str) -> dict[str, Any]:
    raw = str(raw_sql or "").strip()
    lower = raw.lower()
    is_select = lower.startswith(("select", "with", "show", "explain"))
    return {"raw": raw, "is_select": is_select}
