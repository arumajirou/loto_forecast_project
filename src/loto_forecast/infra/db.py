from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DEFAULT_DB_PATH = Path.cwd() / "data" / "registry.sqlite"
DB_PATH = os.environ.get("LOTO_DB_PATH", str(DEFAULT_DB_PATH))
Path(DB_PATH).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
event.listen(engine, "connect", _set_sqlite_pragmas)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_session():
    return SessionLocal()


def init_db() -> None:
    # lazy import to avoid circular references
    from loto_forecast.infra.orm_models import Base as _Base  # noqa: F401

    _Base.metadata.create_all(bind=engine)
