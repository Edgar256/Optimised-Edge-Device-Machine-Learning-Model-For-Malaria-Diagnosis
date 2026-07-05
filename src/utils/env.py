"""Environment variable helpers."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from src.utils.paths import find_project_root


def _load_dotenv() -> None:
    load_dotenv(find_project_root() / ".env", override=False)


def normalize_database_url(url: str) -> str:
    """Convert shorthand mysql:// URLs to SQLAlchemy-compatible drivers."""
    if url.startswith("mysql://"):
        return "mysql+pymysql://" + url[len("mysql://") :]
    return url


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
