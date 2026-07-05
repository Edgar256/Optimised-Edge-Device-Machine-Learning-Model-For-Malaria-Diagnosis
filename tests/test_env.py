"""Tests for environment helpers."""

from src.utils.env import normalize_database_url


def test_normalize_database_url_converts_mysql_scheme() -> None:
    url = normalize_database_url("mysql://user:pass@localhost:3306/malaria")
    assert url.startswith("mysql+pymysql://")
    assert url.endswith("/malaria")


def test_normalize_database_url_strips_connection_limit() -> None:
    url = normalize_database_url(
        "mysql://user:pass@host.example.com:3306/db?connection_limit=5&charset=utf8mb4"
    )
    assert "connection_limit" not in url
    assert "charset=utf8mb4" in url
