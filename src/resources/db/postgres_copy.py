from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)
from sqlalchemy import BigInteger, Boolean, DateTime, Float, Text, inspect, text
from sqlalchemy.engine import Engine

from ..utils import safe_ident

NULL_TOKEN = "\\N"
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
CSV_QUOTE_CHARS = (",", '"', "\n", "\r")
COPY_COLUMNS_KEY = "__columns__"
COPY_ARRAYS_KEY = "__arrays__"
DEFAULT_COPY_STRATEGY = "csv_buffer"


def quote_ident(value: str) -> str:
    return f'"{str(value).replace(chr(34), chr(34) * 2)}"'


def table_ref(schema: str, table: str) -> str:
    return f'{quote_ident(safe_ident(schema))}.{quote_ident(safe_ident(table))}'


def _sqlalchemy_type_for_series(series: pd.Series) -> Any:
    if is_bool_dtype(series.dtype):
        return Boolean()
    if is_integer_dtype(series.dtype):
        return BigInteger()
    if is_float_dtype(series.dtype):
        return Float(precision=53)
    if is_datetime64_any_dtype(series.dtype):
        tz = getattr(series.dtype, "tz", None) is not None
        return DateTime(timezone=tz)
    return Text()


def _create_table_from_columns(engine: Engine, payload: dict[str, Any], schema: str, table: str) -> None:
    col_defs: list[str] = []
    for column in payload["columns"]:
        compiled = _sqlalchemy_type_for_series(pd.Series(payload["arrays"][column], copy=False)).compile(dialect=engine.dialect)
        col_defs.append(f"{quote_ident(column)} {compiled}")
    ddl = f"CREATE TABLE {table_ref(schema, table)} ({', '.join(col_defs)})"
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _fetch_existing_columns(engine: Engine, schema: str, table: str) -> list[str]:
    sql = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": safe_ident(schema), "table": safe_ident(table)}).fetchall()
    return [str(row[0]) for row in rows]


def _normalize_copy_payload(payload: pd.DataFrame | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, pd.DataFrame):
        columns = tuple(str(c) for c in payload.columns)
        arrays = {column: payload[column].to_numpy(copy=False) for column in columns}
        return {"columns": columns, "arrays": arrays, "rows": int(len(payload))}
    if not isinstance(payload, Mapping):
        raise TypeError(f"unsupported copy payload type: {type(payload)!r}")

    if COPY_COLUMNS_KEY in payload or COPY_ARRAYS_KEY in payload:
        columns = tuple(str(c) for c in payload.get(COPY_COLUMNS_KEY, ()))
        arrays_input = payload.get(COPY_ARRAYS_KEY, {})
        if not isinstance(arrays_input, Mapping):
            raise TypeError("copy payload __arrays__ must be a mapping")
        arrays = {str(column): np.asarray(values) for column, values in arrays_input.items()}
    else:
        columns = tuple(str(c) for c in payload)
        arrays = {column: np.asarray(payload[column]) for column in columns}

    rows: int | None = None
    for column in columns:
        arr = arrays.get(column)
        if arr is None:
            raise ValueError(f"copy payload missing array for column: {column}")
        if arr.ndim != 1:
            raise ValueError(f"copy payload column must be 1-D: {column}")
        if rows is None:
            rows = int(len(arr))
        elif int(len(arr)) != rows:
            raise ValueError(f"copy payload column length mismatch: {column}")
    return {"columns": columns, "arrays": arrays, "rows": int(rows or 0)}


def _make_null_array(length: int) -> np.ndarray:
    out = np.empty(length, dtype=object)
    out.fill(pd.NA)
    return out


def _prepare_payload(engine: Engine, payload: dict[str, Any], schema: str, table: str, if_exists: str) -> dict[str, Any]:
    mode = str(if_exists).strip().lower()
    if mode not in {"replace", "append", "fail"}:
        raise ValueError(f"unsupported if_exists: {if_exists}")

    safe_schema = safe_ident(schema)
    safe_table = safe_ident(table)
    insp = inspect(engine)
    exists = insp.has_table(safe_table, schema=safe_schema)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(safe_schema)}"))

    if exists and mode == "fail":
        raise ValueError(f"target table already exists: {safe_schema}.{safe_table}")

    if exists and mode == "replace":
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {table_ref(safe_schema, safe_table)}"))
        exists = False

    if not exists:
        _create_table_from_columns(engine, payload, safe_schema, safe_table)
        return payload

    existing_columns = _fetch_existing_columns(engine, safe_schema, safe_table)
    extra = [column for column in payload["columns"] if column not in existing_columns]
    if extra:
        raise ValueError(f"append target {safe_schema}.{safe_table} is missing columns for payload: {extra}")

    arrays = dict(payload["arrays"])
    for column in existing_columns:
        if column not in arrays:
            arrays[column] = _make_null_array(payload["rows"])
    return {"columns": tuple(existing_columns), "arrays": arrays, "rows": payload["rows"]}


def _slice_payload(payload: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    arrays = {column: payload["arrays"][column][start:end] for column in payload["columns"]}
    return {"columns": payload["columns"], "arrays": arrays, "rows": max(0, int(end - start))}


def _csv_escape(value: str) -> str:
    if any(ch in value for ch in CSV_QUOTE_CHARS):
        return '"' + value.replace('"', '""') + '"'
    return value


def _format_datetime_csv(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        if value is pd.NaT:
            return NULL_TOKEN
        return value.strftime(DATETIME_FORMAT)
    if isinstance(value, datetime):
        return value.strftime(DATETIME_FORMAT)
    return _csv_escape(str(value))


def _postgres_copy_type_for_array(values: np.ndarray) -> str:
    if values.dtype.kind in {"i", "u"}:
        return "int8"
    if values.dtype.kind == "f":
        return "float8"
    if values.dtype.kind == "b":
        return "bool"
    if is_datetime64_any_dtype(values.dtype):
        tz = getattr(values.dtype, "tz", None)
        return "timestamptz" if tz is not None else "timestamp"
    if values.dtype.kind == "M":
        return "timestamp"
    if values.dtype.kind != "O":
        return "text"

    for value in values:
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            if value is pd.NaT:
                continue
            return "timestamptz" if value.tzinfo is not None else "timestamp"
        if isinstance(value, datetime):
            return "timestamptz" if value.tzinfo is not None else "timestamp"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int8"
        if isinstance(value, float):
            return "float8"
        return "text"
    return "text"


def _serialize_object_array(values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, float]:
    import time

    out = np.empty(len(values), dtype=object)
    t_null = time.perf_counter()
    out[:] = NULL_TOKEN
    null_token_emit = time.perf_counter() - t_null
    for idx in np.flatnonzero(valid_mask):
        value = values[idx]
        if isinstance(value, (pd.Timestamp, datetime)):
            out[idx] = _format_datetime_csv(value)
        else:
            out[idx] = _csv_escape(str(value))
    return out, null_token_emit


def _serialize_datetime_array(values: np.ndarray, null_mask: np.ndarray) -> tuple[np.ndarray, float]:
    import time

    out = np.empty(len(values), dtype=object)
    t_null = time.perf_counter()
    out[:] = NULL_TOKEN
    null_token_emit = time.perf_counter() - t_null
    valid_idx = np.flatnonzero(~null_mask)
    if valid_idx.size == 0:
        return out, null_token_emit
    formatted = pd.to_datetime(values[valid_idx]).strftime(DATETIME_FORMAT)
    out[valid_idx] = formatted.to_numpy(dtype=object, copy=False)
    return out, null_token_emit


def _serialize_float_array(values: np.ndarray, null_mask: np.ndarray) -> tuple[np.ndarray, float]:
    import time

    out = np.empty(len(values), dtype=object)
    t_null = time.perf_counter()
    out[:] = NULL_TOKEN
    null_token_emit = time.perf_counter() - t_null
    valid_idx = np.flatnonzero(~null_mask)
    if valid_idx.size == 0:
        return out, null_token_emit
    fmt = "%.9g" if values.dtype.itemsize <= 4 else "%.17g"
    out[valid_idx] = np.char.mod(fmt, values[valid_idx]).astype(object, copy=False)
    return out, null_token_emit


def _serialize_column(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    import time

    metrics = {
        "null_convert": 0.0,
        "scalar_format": 0.0,
        "float_format": 0.0,
        "datetime_format": 0.0,
        "object_format": 0.0,
        "null_token_emit": 0.0,
    }
    t_null = time.perf_counter()
    if values.dtype.kind == "f":
        null_mask = np.isnan(values)
    elif values.dtype.kind in {"i", "u", "b"}:
        null_mask = np.zeros(len(values), dtype=bool)
    else:
        null_mask = pd.isna(values)
    metrics["null_convert"] = time.perf_counter() - t_null

    if values.dtype.kind == "f":
        t_format = time.perf_counter()
        formatted, null_token_emit = _serialize_float_array(values, null_mask)
        metrics["float_format"] = time.perf_counter() - t_format
        metrics["null_token_emit"] = float(null_token_emit)
    elif values.dtype.kind in {"i", "u"}:
        t_format = time.perf_counter()
        formatted = values.astype(str).astype(object, copy=False)
        metrics["object_format"] = time.perf_counter() - t_format
    elif values.dtype.kind == "b":
        t_format = time.perf_counter()
        formatted = np.where(values, "True", "False").astype(object, copy=False)
        metrics["object_format"] = time.perf_counter() - t_format
    elif is_datetime64_any_dtype(values.dtype):
        t_format = time.perf_counter()
        formatted, null_token_emit = _serialize_datetime_array(values, null_mask)
        metrics["datetime_format"] = time.perf_counter() - t_format
        metrics["null_token_emit"] = float(null_token_emit)
    else:
        t_format = time.perf_counter()
        formatted, null_token_emit = _serialize_object_array(values, ~null_mask)
        metrics["object_format"] = time.perf_counter() - t_format
        metrics["null_token_emit"] = float(null_token_emit)

    metrics["scalar_format"] = (
        metrics["float_format"] + metrics["datetime_format"] + metrics["object_format"] + metrics["null_token_emit"]
    )
    return formatted, metrics


def _payload_to_csv_buffer(payload: dict[str, Any]) -> tuple[io.StringIO, dict[str, float]]:
    import time

    formatted_columns: list[np.ndarray] = []
    metrics = {
        "null_convert": 0.0,
        "scalar_format": 0.0,
        "float_format": 0.0,
        "datetime_format": 0.0,
        "object_format": 0.0,
        "null_token_emit": 0.0,
    }
    for column in payload["columns"]:
        formatted, column_metrics = _serialize_column(payload["arrays"][column])
        formatted_columns.append(formatted)
        for key, value in column_metrics.items():
            metrics[key] += float(value)

    t_rows = time.perf_counter()
    rows = [",".join(row) for row in zip(*formatted_columns, strict=False)] if formatted_columns else []
    row_serialize_sec = time.perf_counter() - t_rows

    t_join = time.perf_counter()
    data = "\n".join(rows)
    if data:
        data += "\n"
    buffer_join_sec = time.perf_counter() - t_join
    buf = io.StringIO(data)
    buf.seek(0)
    return buf, {
        "copy_prepare": metrics["null_convert"] + metrics["scalar_format"] + row_serialize_sec + buffer_join_sec,
        "null_convert": metrics["null_convert"],
        "scalar_format": metrics["scalar_format"],
        "float_format": metrics["float_format"],
        "datetime_format": metrics["datetime_format"],
        "object_format": metrics["object_format"],
        "null_token_emit": metrics["null_token_emit"],
        "row_serialize": row_serialize_sec,
        "buffer_join": buffer_join_sec,
    }


def _coerce_copy_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        if value is pd.NaT:
            return None
        return value.to_pydatetime()
    return value


def _iter_copy_rows(payload: dict[str, Any]) -> Iterator[tuple[Any, ...]]:
    arrays = [payload["arrays"][column] for column in payload["columns"]]
    total_rows = int(payload["rows"])
    for row_idx in range(total_rows):
        yield tuple(_coerce_copy_value(arr[row_idx]) for arr in arrays)


def _copy_chunk_csv(cursor: Any, chunk: dict[str, Any], schema: str, table: str) -> dict[str, float]:
    import time

    columns = [str(c) for c in chunk["columns"]]
    sql = (
        f"COPY {table_ref(schema, table)} "
        f"({', '.join(quote_ident(c) for c in columns)}) "
        "FROM STDIN WITH (FORMAT CSV, HEADER FALSE, NULL '\\N')"
    )
    buf, timing = _payload_to_csv_buffer(chunk)
    t_execute = time.perf_counter()
    cursor.copy_expert(sql, buf)
    timing["copy_execute"] = time.perf_counter() - t_execute
    return timing


def _make_psycopg_conninfo(engine: Engine) -> str:
    url = engine.url
    parts = []
    if url.host:
        parts.append(f"host={url.host}")
    if url.port:
        parts.append(f"port={url.port}")
    if url.database:
        parts.append(f"dbname={url.database}")
    if url.username:
        parts.append(f"user={url.username}")
    if url.password:
        parts.append(f"password={url.password}")
    return " ".join(parts)


def _copy_payload_psycopg3(
    engine: Engine,
    payload: dict[str, Any],
    schema: str,
    table: str,
    chunk_rows: int,
    *,
    binary: bool,
) -> dict[str, float]:
    import time

    conninfo = _make_psycopg_conninfo(engine)
    columns = [str(c) for c in payload["columns"]]
    sql = (
        f"COPY {table_ref(schema, table)} "
        f"({', '.join(quote_ident(c) for c in columns)}) "
        f"FROM STDIN (FORMAT {'BINARY' if binary else 'TEXT'})"
    )
    column_types = [_postgres_copy_type_for_array(payload["arrays"][column]) for column in payload["columns"]]
    total_rows = int(payload["rows"])
    chunk_size = max(1, int(chunk_rows))
    totals = {
        "copy_prepare_sec": 0.0,
        "null_convert_sec": 0.0,
        "scalar_format_sec": 0.0,
        "float_format_sec": 0.0,
        "datetime_format_sec": 0.0,
        "object_format_sec": 0.0,
        "null_token_emit_sec": 0.0,
        "row_serialize_sec": 0.0,
        "buffer_join_sec": 0.0,
        "copy_execute_sec": 0.0,
    }
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cur:
            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                chunk = _slice_payload(payload, start, end)
                t_prepare = time.perf_counter()
                row_iter = _iter_copy_rows(chunk)
                totals["copy_prepare_sec"] += time.perf_counter() - t_prepare
                t_execute = time.perf_counter()
                with cur.copy(sql) as copy:
                    if binary:
                        copy.set_types(column_types)
                    for row in row_iter:
                        copy.write_row(row)
                totals["copy_execute_sec"] += time.perf_counter() - t_execute
        conn.commit()
    return totals


def copy_dataframe_to_postgres(
    engine: Engine,
    df: pd.DataFrame | Mapping[str, Any],
    *,
    schema: str,
    table: str,
    if_exists: str = "append",
    chunk_rows: int = 50000,
    copy_strategy: str = DEFAULT_COPY_STRATEGY,
) -> dict[str, Any]:
    safe_schema = safe_ident(schema)
    safe_table = safe_ident(table)
    payload = _normalize_copy_payload(df)
    aligned = _prepare_payload(engine, payload, safe_schema, safe_table, if_exists=if_exists)
    total_rows = int(aligned["rows"])

    if total_rows == 0:
        return {"schema": safe_schema, "table": safe_table, "rows": 0, "chunks": 0}

    strategy = str(copy_strategy).strip().lower()
    if strategy not in {"csv_buffer", "psycopg3_row", "binary_copy"}:
        raise ValueError(f"unsupported copy_strategy: {copy_strategy}")

    if strategy in {"psycopg3_row", "binary_copy"}:
        totals = _copy_payload_psycopg3(
            engine,
            aligned,
            safe_schema,
            safe_table,
            chunk_rows,
            binary=(strategy == "binary_copy"),
        )
        chunks = (total_rows + max(1, int(chunk_rows)) - 1) // max(1, int(chunk_rows))
        return {"schema": safe_schema, "table": safe_table, "rows": total_rows, "chunks": chunks, **totals}

    chunk_size = max(1, int(chunk_rows))
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        try:
            chunks = 0
            totals = {
                "copy_prepare_sec": 0.0,
                "null_convert_sec": 0.0,
                "scalar_format_sec": 0.0,
                "float_format_sec": 0.0,
                "datetime_format_sec": 0.0,
                "object_format_sec": 0.0,
                "null_token_emit_sec": 0.0,
                "row_serialize_sec": 0.0,
                "buffer_join_sec": 0.0,
                "copy_execute_sec": 0.0,
            }
            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                timing = _copy_chunk_csv(cursor, _slice_payload(aligned, start, end), safe_schema, safe_table)
                totals["copy_prepare_sec"] += float(timing.get("copy_prepare", 0.0))
                totals["null_convert_sec"] += float(timing.get("null_convert", 0.0))
                totals["scalar_format_sec"] += float(timing.get("scalar_format", 0.0))
                totals["float_format_sec"] += float(timing.get("float_format", 0.0))
                totals["datetime_format_sec"] += float(timing.get("datetime_format", 0.0))
                totals["object_format_sec"] += float(timing.get("object_format", 0.0))
                totals["null_token_emit_sec"] += float(timing.get("null_token_emit", 0.0))
                totals["row_serialize_sec"] += float(timing.get("row_serialize", 0.0))
                totals["buffer_join_sec"] += float(timing.get("buffer_join", 0.0))
                totals["copy_execute_sec"] += float(timing.get("copy_execute", 0.0))
                chunks += 1
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()

    return {"schema": safe_schema, "table": safe_table, "rows": total_rows, "chunks": chunks, **totals}
