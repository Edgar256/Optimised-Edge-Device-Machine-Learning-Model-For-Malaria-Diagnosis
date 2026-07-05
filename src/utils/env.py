"""Environment variable helpers."""

from __future__ import annotations

import os
import ssl
from functools import lru_cache

from dotenv import load_dotenv

from src.utils.paths import find_project_root

# Query params handled via connect_args instead of the SQLAlchemy URL.
_MYSQL_SSL_QUERY_KEYS = frozenset({"ssl", "ssl_mode", "ssl_verify"})

# Hosted DB URLs may include pool hints that pymysql.connect() rejects.
_MYSQL_UNSUPPORTED_QUERY_KEYS = frozenset(
    {
        "connection_limit",
        "pool_timeout",
        "pgbouncer",
        "sslaccept",
        *_MYSQL_SSL_QUERY_KEYS,
    }
)


def _load_dotenv() -> None:
    load_dotenv(find_project_root() / ".env", override=False)


def _parse_mysql_url(url: str):
    from sqlalchemy.engine import make_url

    if url.startswith("mysql://"):
        url = "mysql+pymysql://" + url[len("mysql://") :]
    return make_url(url)


def _mysql_ssl_verify(query: dict[str, str]) -> bool:
    ssl_verify = query.get("ssl_verify", "true").lower()
    return ssl_verify not in {"false", "0", "no"}


def _mysql_ssl_enabled(query: dict[str, str]) -> bool:
    ssl_value = query.get("ssl", "").lower()
    ssl_mode = query.get("ssl_mode", "").lower()
    if ssl_value in {"true", "1", "yes"}:
        return True
    return ssl_mode in {"required", "verify_ca", "verify_identity"}


def _build_mysql_ssl_context(query: dict[str, str]) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if not _mysql_ssl_verify(query):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def normalize_database_url(url: str) -> str:
    """Convert shorthand mysql:// URLs and drop pymysql-incompatible query params."""
    parsed = _parse_mysql_url(url)
    driver = parsed.drivername.split("+", 1)[0]
    if driver != "mysql":
        return url

    if parsed.query:
        filtered = {
            key: value
            for key, value in parsed.query.items()
            if key not in _MYSQL_UNSUPPORTED_QUERY_KEYS
        }
        parsed = parsed.set(query=filtered)
    return str(parsed)


def get_mysql_connect_args(url: str | None = None) -> dict:
    """Return pymysql connect_args (e.g. SSL) derived from DATABASE_URL query params."""
    raw = (url or os.getenv("DATABASE_URL", "")).strip()
    if not raw.startswith(("mysql://", "mysql+pymysql://")):
        return {}

    parsed = _parse_mysql_url(raw)
    if parsed.drivername.split("+", 1)[0] != "mysql":
        return {}

    query = dict(parsed.query or {})
    if _mysql_ssl_enabled(query):
        return {"ssl": _build_mysql_ssl_context(query)}
    return {}


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
    port = parsed.port or 3306
    return f"user={username} host={host} port={port} database={database}"


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
