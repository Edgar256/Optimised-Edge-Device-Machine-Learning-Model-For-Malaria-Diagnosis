"""Environment variable helpers."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from src.utils.paths import find_project_root


def _load_dotenv() -> None:
    load_dotenv(find_project_root() / ".env", override=False)


def normalize_database_url(url: str) -> str:
    """Convert shorthand mysql:// URLs and drop pymysql-incompatible query params."""
    from sqlalchemy.engine import make_url

    if url.startswith("mysql://"):
        url = "mysql+pymysql://" + url[len("mysql://") :]

    parsed = make_url(url)
    driver = parsed.drivername.split("+", 1)[0]
    if driver != "mysql":
        return url

    # Hosted DB URLs (Render, PlanetScale, etc.) may include pool hints that
    # SQLAlchemy forwards to pymysql.connect(), which rejects them.
    unsupported = frozenset(
        {
            "connection_limit",
            "pool_timeout",
            "pgbouncer",
            "sslaccept",
        }
    )
    if parsed.query:
        filtered = {key: value for key, value in parsed.query.items() if key not in unsupported}
        parsed = parsed.set(query=filtered)
    return str(parsed)


def describe_database_target(url: str | None = None) -> str:
    """Return a log-safe summary of the configured database target."""
    from sqlalchemy.engine import make_url

    raw = url or get_database_url()
    if not raw:
        return "DATABASE_URL is not set"
    parsed = make_url(raw)
    host = parsed.host or "unknown-host"
    if host in {"localhost", "127.0.0.1"}:
        host = f"{host} (will not work on Render — use a hosted MySQL URL)"
    database = parsed.database or "unknown-database"
    username = parsed.username or "unknown-user"
    return f"user={username} host={host} database={database}"


@lru_cache(maxsize=1)
def get_database_url() -> str | None:
    _load_dotenv()
    value = os.getenv("DATABASE_URL", "").strip()
    return normalize_database_url(value) if value else None


@lru_cache(maxsize=1)
def get_jwt_secret() -> str:
    _load_dotenv()
    return os.getenv("JWT_SECRET", "dev-only-change-me").strip() or "dev-only-change-me"


@lru_cache(maxsize=1)
def get_api_url() -> str | None:
    _load_dotenv()
    value = os.getenv("API_URL", "").strip()
    return value or None


@lru_cache(maxsize=1)
def get_admin_registration_secret() -> str | None:
    _load_dotenv()
    value = os.getenv("ADMIN_REGISTRATION_SECRET", "").strip()
    return value or None


def reset_env_cache() -> None:
    get_database_url.cache_clear()
    get_jwt_secret.cache_clear()
    get_api_url.cache_clear()
    get_admin_registration_secret.cache_clear()
