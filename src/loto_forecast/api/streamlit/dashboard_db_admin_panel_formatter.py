from __future__ import annotations

import pandas as pd


def build_table_options(table_map: dict[str, list[str]]) -> dict[str, list[str] | str | None]:
    if not table_map:
        return {"schemas": [], "selected_schema": None, "tables": [], "selected_table": None}
    schemas = sorted(table_map.keys())
    selected_schema = schemas[0]
    tables = table_map[selected_schema]
    selected_table = tables[0] if tables else None
    return {
        "schemas": schemas,
        "selected_schema": selected_schema,
        "tables": tables,
        "selected_table": selected_table,
    }


def build_columns_caption(row_columns: list[str]) -> str | None:
    if not row_columns:
        return None
    return "columns: " + ", ".join(row_columns)


def default_er_schemas(table_df: pd.DataFrame) -> list[str]:
    if table_df.empty or "table_schema" not in table_df.columns:
        return []
    schema_opts = sorted(table_df["table_schema"].astype(str).unique().tolist())
    return [s for s in ["dataset", "exog", "resources", "meta", "model"] if s in schema_opts] or schema_opts[:5]


def filter_fk_rows(fk_df: pd.DataFrame, *, selected_schemas: list[str]) -> pd.DataFrame:
    if fk_df.empty or not selected_schemas:
        return pd.DataFrame()
    return fk_df[
        fk_df["src_schema"].astype(str).isin(selected_schemas)
        & fk_df["ref_schema"].astype(str).isin(selected_schemas)
    ].copy()
