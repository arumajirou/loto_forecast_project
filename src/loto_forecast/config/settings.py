from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _parse_int_list(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return fallback
    vals: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        vals.append(int(token))
    return vals or fallback


class Settings(BaseModel):
    # DB
    db_host: str = os.getenv("DB_HOST", "127.0.0.1")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_user: str = os.getenv("DB_USER", "loto")
    db_password: str = os.getenv("DB_PASSWORD", "")
    db_name: str = os.getenv("DB_NAME", "loto")
    db_schema: str = os.getenv("DB_SCHEMA", "dataset")
    db_table: str = os.getenv("DB_TABLE", "loto_y_ts")
    meta_schema: str = os.getenv("META_SCHEMA", "meta")
    meta_table: str = os.getenv("META_TABLE", "nf_automodel")
    model_schema: str = os.getenv("MODEL_SCHEMA", "model")
    model_table: str = os.getenv("MODEL_TABLE", "nf_automodel")
    exog_schema: str = os.getenv("EXOG_SCHEMA", "exog")
    resources_schema: str = os.getenv("RESOURCES_SCHEMA", "resources")
    catalog_schema: str = os.getenv("CATALOG_SCHEMA", "catalog")
    log_schema: str = os.getenv("LOG_SCHEMA", "log")

    # run
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "./artifacts"))
    log_dir: Path = Path(os.getenv("LOG_DIR", "./logs"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # time series
    freq: str = os.getenv("FREQ", "D")
    default_horizon: int = int(os.getenv("DEFAULT_HORIZON", "28"))
    time_col: str = os.getenv("TIME_COL", "ds")
    target_col: str = os.getenv("TARGET_COL", "y")
    id_col: str = os.getenv("ID_COL", "unique_id")
    default_lags: list[int] = _parse_int_list(os.getenv("DEFAULT_LAGS"), [1, 7, 14])
    default_windows: list[int] = _parse_int_list(os.getenv("DEFAULT_WINDOWS"), [7, 14, 28])

    # catalog/codegen
    codegen_yaml_path: str = os.getenv(
        "CODEGEN_YAML_PATH",
        "./docs/lib_docs/neuralforecast_all_codegen.yaml",
    )


settings = Settings()
