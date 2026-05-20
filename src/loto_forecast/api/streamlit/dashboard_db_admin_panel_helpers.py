from __future__ import annotations

from typing import Any


def validate_confirmation(expected: str, actual: str) -> None:
    if str(actual).strip() != str(expected):
        raise ValueError("confirmation text mismatch")


def build_create_table_sql(
    schema: str,
    table_name: str,
    columns: list[dict[str, Any]],
    *,
    safe_ident: Any,
    normalize_type: Any,
) -> str:
    table_ident = safe_ident(str(table_name).strip())
    col_defs: list[str] = []
    pk_cols: list[str] = []
    for row in columns:
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        col_ident = safe_ident(name)
        col_type = normalize_type(str(row.get("type") or ""))
        nullable = bool(row.get("nullable", True))
        unique = bool(row.get("unique", False))
        is_pk = bool(row.get("primary_key", False))
        parts = [col_ident, col_type]
        if not nullable:
            parts.append("NOT NULL")
        if unique and not is_pk:
            parts.append("UNIQUE")
        col_defs.append(" ".join(parts))
        if is_pk:
            pk_cols.append(col_ident)
    if not col_defs:
        raise ValueError("at least one valid column is required")
    if pk_cols:
        col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
    return f"CREATE TABLE {safe_ident(schema)}.{table_ident} (" + ", ".join(col_defs) + ")"


def build_add_column_sql(
    schema: str,
    table: str,
    column_name: str,
    column_type: str,
    *,
    nullable: bool,
    safe_ident: Any,
    normalize_type: Any,
) -> str:
    sql = (
        f"ALTER TABLE {safe_ident(schema)}.{safe_ident(table)} "
        f"ADD COLUMN {safe_ident(column_name.strip())} {normalize_type(column_type)}"
    )
    if not nullable:
        sql += " NOT NULL"
    return sql


def build_insert_sql(
    schema: str,
    table: str,
    payload: dict[str, Any],
    *,
    safe_ident: Any,
) -> tuple[str, dict[str, Any]]:
    cols = list(payload.keys())
    col_sql = ", ".join(safe_ident(c) for c in cols)
    val_sql = ", ".join(f":v_{i}" for i in range(len(cols)))
    params = {f"v_{i}": payload[col] for i, col in enumerate(cols)}
    sql = f"INSERT INTO {safe_ident(schema)}.{safe_ident(table)} ({col_sql}) VALUES ({val_sql})"
    return sql, params


def build_select_sql(
    schema: str,
    table: str,
    where_payload: dict[str, Any],
    *,
    limit: int,
    safe_ident: Any,
    build_where_equals_clause: Any,
) -> tuple[str, dict[str, Any]]:
    where_sql, where_params = build_where_equals_clause(where_payload, param_prefix="rw")
    sql = f"SELECT * FROM {safe_ident(schema)}.{safe_ident(table)} WHERE {where_sql} LIMIT {int(limit)}"
    return sql, where_params


def build_update_sql(
    schema: str,
    table: str,
    set_payload: dict[str, Any],
    where_payload: dict[str, Any],
    *,
    safe_ident: Any,
    build_where_equals_clause: Any,
) -> tuple[str, dict[str, Any]]:
    set_parts: list[str] = []
    set_params: dict[str, Any] = {}
    for i, (col, val) in enumerate(set_payload.items()):
        pkey = f"sv_{i}"
        set_parts.append(f"{safe_ident(col)} = :{pkey}")
        set_params[pkey] = val
    where_sql, where_params = build_where_equals_clause(where_payload, param_prefix="uw")
    sql = f"UPDATE {safe_ident(schema)}.{safe_ident(table)} SET {', '.join(set_parts)} WHERE {where_sql}"
    return sql, {**set_params, **where_params}


def build_delete_sql(
    schema: str,
    table: str,
    where_payload: dict[str, Any],
    *,
    safe_ident: Any,
    build_where_equals_clause: Any,
) -> tuple[str, dict[str, Any]]:
    where_sql, where_params = build_where_equals_clause(where_payload, param_prefix="dw")
    sql = f"DELETE FROM {safe_ident(schema)}.{safe_ident(table)} WHERE {where_sql}"
    return sql, where_params
