"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.base import Base
from src.utils.env import get_database_url, get_mysql_connect_args

_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        database_url = get_database_url()
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to .env (see .env.example)."
            )
        engine_kwargs: dict = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            if database_url.endswith(":memory:") or database_url.rstrip("/").endswith(":memory:"):
                engine_kwargs["poolclass"] = StaticPool
        else:
            mysql_connect_args = get_mysql_connect_args()
            if mysql_connect_args:
                engine_kwargs["connect_args"] = mysql_connect_args
        _engine = create_engine(database_url, **engine_kwargs)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    _get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def init_db() -> list[str]:
    """Create database tables and apply additive migrations."""
    from src.database import models  # noqa: F401 — register models
    from src.database.migrations import migrate_schema

    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    return migrate_schema(engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Clear cached engine (used in tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    from src.utils.env import reset_env_cache

    reset_env_cache()
