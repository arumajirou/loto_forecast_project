#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "data_jobs"
DEFAULT_SOURCE_SCHEMA = "dataset"
DEFAULT_SOURCE_TABLE = "loto_y_ts_unified"
DEFAULT_TARGET_SCHEMA = "exog"
DEFAULT_TARGET_TABLE = "nf_feature_table_auto"

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DANGEROUS_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|vacuum|copy|merge)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class JobManifest:
    run_id: str
    started_at: str
    finished_at: str | None
    mode: str
    source: dict[str, Any]
    target: dict[str, Any]
    rows_read: int
    rows_generated: int
    columns_generated: list[str]
    wrote_db: bool
    artifact_dir: str
    warnings: list[str]
    status: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_ident(value: str) -> str:
    if not IDENT_RE.match(value):
        raise ValueError(f"unsafe identifier: {value!r}")
    return f'"{value}"'


def ensure_select_only(sql: str) -> str:
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^(select|with)\b", stripped, flags=re.IGNORECASE):
        raise ValueError("source SQL must start with SELECT or WITH")
    if DANGEROUS_SQL_RE.search(stripped):
        raise ValueError("source SQL must be read-only and cannot contain DDL/DML keywords")
    return stripped


def build_engine() -> Any:
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv(PROJECT_ROOT / ".env")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "5432"))
    user = os.getenv("DB_USER", "loto")
    database = os.getenv("DB_NAME", "loto")
    password = os.getenv("DB_PASSWORD", "")
    if not password:
        raise RuntimeError("DB_PASSWORD is not set. Put it in .env or environment variables; never pass it as a CLI argument.")
    dsn = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    return create_engine(dsn, pool_pre_ping=True)


def load_source_dataframe(engine: Any, args: argparse.Namespace) -> Any:
    import pandas as pd
    from sqlalchemy import text

    if args.source_csv:
        path = Path(args.source_csv).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return pd.read_csv(path)

    if args.source_sql:
        sql = ensure_select_only(args.source_sql)
    else:
        sql = (
            f"SELECT * FROM {safe_ident(args.source_schema)}.{safe_ident(args.source_table)} "
            f"ORDER BY {safe_ident(args.time_col)} LIMIT :limit"
        )
    return pd.read_sql_query(text(sql), engine, params={"limit": int(args.limit)})


def generate_features(df: Any, *, id_col: str, time_col: str, target_col: str) -> Any:
    import pandas as pd

    if df.empty:
        return df.copy()

    out = df.copy()
    warnings: list[str] = []
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out["ds_year"] = out[time_col].dt.year
        out["ds_month"] = out[time_col].dt.month
        out["ds_day"] = out[time_col].dt.day
        out["ds_dayofweek"] = out[time_col].dt.dayofweek
        out["ds_is_month_start"] = out[time_col].dt.is_month_start.astype("Int64")
        out["ds_is_month_end"] = out[time_col].dt.is_month_end.astype("Int64")
    else:
        warnings.append(f"time column not found: {time_col}")

    if target_col in out.columns:
        sort_cols = [c for c in (id_col, time_col) if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols)
        group_key = out[id_col] if id_col in out.columns else pd.Series(["__all__"] * len(out), index=out.index)
        out["y_lag_1"] = out.groupby(group_key, dropna=False)[target_col].shift(1)
        out["y_lag_2"] = out.groupby(group_key, dropna=False)[target_col].shift(2)
        out["y_lag_3"] = out.groupby(group_key, dropna=False)[target_col].shift(3)
        out["y_rolling_mean_3"] = out.groupby(group_key, dropna=False)[target_col].shift(1).rolling(3, min_periods=1).mean()
        out["y_rolling_std_3"] = out.groupby(group_key, dropna=False)[target_col].shift(1).rolling(3, min_periods=2).std()
        out["y_diff_1"] = out.groupby(group_key, dropna=False)[target_col].diff(1)
    else:
        warnings.append(f"target column not found: {target_col}")

    out.attrs["warnings"] = warnings
    return out


def write_feature_table(engine: Any, df: Any, *, schema: str, table: str, if_exists: str) -> None:
    from sqlalchemy import text

    if schema == "dataset":
        raise RuntimeError("dataset schema is read-only. Choose exog, model, meta, resources, catalog, or log.")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {safe_ident(schema)}"))
    df.to_sql(table, engine, schema=schema, if_exists=if_exists, index=False, method="multi", chunksize=1000)


def write_manifest(manifest: JobManifest) -> None:
    path = Path(manifest.artifact_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch dataset rows, generate basic forecasting features, and optionally write a DB table."
    )
    parser.add_argument("--source-schema", default=DEFAULT_SOURCE_SCHEMA)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--source-sql", default="", help="Read-only SELECT/WITH SQL. DDL/DML is rejected.")
    parser.add_argument("--source-csv", default="", help="Optional CSV source instead of DB source.")
    parser.add_argument("--target-schema", default=DEFAULT_TARGET_SCHEMA)
    parser.add_argument("--target-table", default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--id-col", default="unique_id")
    parser.add_argument("--time-col", default="ds")
    parser.add_argument("--target-col", default="y")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="replace")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--yes-write", action="store_true", help="Actually write the generated table.")
    parser.add_argument("--preview-rows", type=int, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = ARTIFACT_ROOT / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = JobManifest(
        run_id=run_id,
        started_at=utc_now_iso(),
        finished_at=None,
        mode="write" if args.yes_write else "dry-run",
        source={
            "schema": args.source_schema,
            "table": args.source_table,
            "sql_provided": bool(args.source_sql),
            "csv": args.source_csv,
            "limit": int(args.limit),
        },
        target={"schema": args.target_schema, "table": args.target_table, "if_exists": args.if_exists},
        rows_read=0,
        rows_generated=0,
        columns_generated=[],
        wrote_db=False,
        artifact_dir=str(artifact_dir),
        warnings=[],
        status="running",
    )

    try:
        engine = None if args.source_csv else build_engine()
        df = load_source_dataframe(engine, args) if engine is not None else load_source_dataframe(None, args)  # type: ignore[arg-type]
        feature_df = generate_features(df, id_col=args.id_col, time_col=args.time_col, target_col=args.target_col)
        manifest.rows_read = int(len(df))
        manifest.rows_generated = int(len(feature_df))
        manifest.columns_generated = list(map(str, feature_df.columns))
        manifest.warnings.extend(feature_df.attrs.get("warnings", []))

        preview_path = artifact_dir / "feature_preview.csv"
        feature_df.head(max(1, int(args.preview_rows))).to_csv(preview_path, index=False)

        if args.yes_write:
            if os.getenv("LOTO_ALLOW_FEATURE_DB_WRITE") != "1":
                raise RuntimeError("Set LOTO_ALLOW_FEATURE_DB_WRITE=1 to allow DB writes.")
            if engine is None:
                engine = build_engine()
            write_feature_table(
                engine,
                feature_df,
                schema=args.target_schema,
                table=args.target_table,
                if_exists=args.if_exists,
            )
            manifest.wrote_db = True

        manifest.status = "ok"
        return_code = 0
    except Exception as exc:  # noqa: BLE001
        manifest.status = "error"
        manifest.warnings.append(f"{type(exc).__name__}: {exc}")
        return_code = 1
    finally:
        manifest.finished_at = utc_now_iso()
        write_manifest(manifest)
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
