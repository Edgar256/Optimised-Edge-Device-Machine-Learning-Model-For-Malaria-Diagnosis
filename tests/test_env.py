"""Tests for environment helpers."""

from src.utils.env import (
    describe_database_target,
    get_mysql_connect_args,
    normalize_database_url,
)


def test_normalize_database_url_converts_mysql_scheme() -> None:
    url = normalize_database_url("mysql://user:pass@localhost:3306/malaria")
    assert url.startswith("mysql+pymysql://")
    assert url.endswith("/malaria")


def test_normalize_database_url_preserves_password() -> None:
    """SQLAlchemy str(URL) redacts passwords as '***'; we must not use that."""
    secret = "placeholder_password_must_survive_normalize"
    url = normalize_database_url(
        f"mysql://dbuser:{secret}@mysql-example.example.com:16447/malaria_data_db"
        "?ssl-mode=REQUIRED&ssl_verify=false"
    )
    assert "***" not in url
    assert secret in url
    assert "ssl-mode" not in url
    assert "ssl_verify" not in url


def test_normalize_database_url_strips_connection_limit() -> None:
    url = normalize_database_url(
        "mysql://user:pass@host.example.com:3306/db?connection_limit=5&charset=utf8mb4"
    )
    assert "connection_limit" not in url
    assert "charset=utf8mb4" in url


def test_normalize_database_url_strips_ssl_query_params() -> None:
    url = normalize_database_url(
        "mysql://user:pass@host.example.com:3306/db?ssl=true&ssl_mode=REQUIRED&ssl_verify=false"
    )
    assert "ssl=" not in url
    assert "ssl_mode=" not in url
    assert "ssl_verify=" not in url


def test_normalize_database_url_strips_aiven_ssl_mode() -> None:
    url = normalize_database_url(
        "mysql://avnadmin:secret@mysql-example.aivencloud.com:16447/malaria_data_db?ssl-mode=REQUIRED"
    )
    assert "ssl-mode" not in url
    assert "ssl_mode" not in url
    assert url.startswith("mysql+pymysql://")
    assert "malaria_data_db" in url


def test_get_mysql_connect_args_ssl_verify_false() -> None:
    url = "mysql://user:pass@host.example.com:3306/db?ssl=true&ssl_verify=false"
    args = get_mysql_connect_args(url)
    assert "ssl" in args
    assert args["ssl"].verify_mode.name == "CERT_NONE"


def test_get_mysql_connect_args_enables_ssl() -> None:
    url = "mysql://user:pass@host.example.com:3306/db?ssl=true"
    assert "ssl" in get_mysql_connect_args(url)


def test_get_mysql_connect_args_ssl_mode_required() -> None:
    url = "mysql://user:pass@host.example.com:3306/db?ssl_mode=REQUIRED"
    assert "ssl" in get_mysql_connect_args(url)


def test_get_mysql_connect_args_aiven_ssl_mode_hyphen() -> None:
    url = (
        "mysql://avnadmin:secret@mysql-example.aivencloud.com:16447/"
        "malaria_data_db?ssl-mode=REQUIRED"
    )
    args = get_mysql_connect_args(url)
    assert "ssl" in args
    assert args["ssl"].verify_mode.name != "CERT_NONE"


def test_get_mysql_connect_args_no_ssl_by_default() -> None:
    url = "mysql://user:pass@host.example.com:3306/db"
    assert get_mysql_connect_args(url) == {}


def test_describe_database_target_flags_localhost_on_render() -> None:
    summary = describe_database_target("mysql://admin:secret@localhost:3306/malaria_prediction")
    assert "user=admin" in summary
    assert "port=3306" in summary
    assert "localhost" in summary
    assert "will not work on Render" in summary
