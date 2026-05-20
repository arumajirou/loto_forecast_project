from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResourcesConfig:
    db_host: str = "127.0.0.1"
    db_port: int = 5432
    db_user: str = "loto"
    db_password: str = ""
    db_name: str = "loto"
    schema: str = "exog"
    namespace: str = "timesfm"
    table_naming: str = "plain"
    app_name: str = "timesfm_exog"
    env: str = "LOCAL"
    profile: str = "local"
    command: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    enable_gpu: bool = True
    enable_sampling: bool = False
    sampling_interval_sec: float = 1.0
    sample_buffer_size: int = 200
    ensure_schema: bool = True
    parallel_snapshot_workers: int = 4

    @classmethod
    def from_env(cls) -> ResourcesConfig:
        return cls(
            db_host=os.getenv("RESMON_DB_HOST", os.getenv("TIMESFM_DB_HOST", "127.0.0.1")),
            db_port=int(os.getenv("RESMON_DB_PORT", os.getenv("TIMESFM_DB_PORT", "5432"))),
            db_user=os.getenv("RESMON_DB_USER", os.getenv("TIMESFM_DB_USER", "loto")),
            db_password=os.getenv(
                "RESMON_DB_PASSWORD",
                os.getenv("TIMESFM_DB_PASSWORD", os.getenv("PGPASSWORD", "")),
            ),
            db_name=os.getenv("RESMON_DB_NAME", os.getenv("TIMESFM_DB_NAME", "loto")),
            schema=os.getenv("RESMON_SCHEMA", "exog"),
            namespace=os.getenv("RESMON_NAMESPACE", "timesfm"),
            table_naming=os.getenv("RESMON_TABLE_NAMING", os.getenv("TIMESFM_RESOURCE_TABLE_NAMING", "plain")),
            app_name=os.getenv("RESMON_APP_NAME", "timesfm_exog"),
            env=os.getenv("RESMON_ENV", "LOCAL"),
            profile=os.getenv("RESMON_PROFILE", "local"),
            enable_gpu=os.getenv("RESMON_ENABLE_GPU", "true").lower() in {"1", "true", "yes"},
            enable_sampling=os.getenv("RESMON_ENABLE_SAMPLING", "false").lower() in {"1", "true", "yes"},
            sampling_interval_sec=float(os.getenv("RESMON_SAMPLING_INTERVAL_SEC", "1.0")),
        )

    def to_dsn(self) -> str:
        return (
            f"host={self.db_host} port={self.db_port} user={self.db_user} "
            f"dbname={self.db_name} password={self.db_password}"
        )

    def validate(self) -> None:
        if not self.namespace.replace("_", "").isalnum():
            raise ValueError("namespace must be alnum/underscore")
        if not self.schema.replace("_", "").isalnum():
            raise ValueError("schema must be alnum/underscore")
        if self.table_naming not in {"plain", "namespaced"}:
            raise ValueError("table_naming must be 'plain' or 'namespaced'")
        if self.parallel_snapshot_workers <= 0:
            raise ValueError("parallel_snapshot_workers must be > 0")
