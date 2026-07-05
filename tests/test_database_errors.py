"""Tests for database error helpers."""

from sqlalchemy.exc import OperationalError

from src.database.errors import (
    database_startup_error_message,
    format_operational_error,
    operational_error_tips,
)


def test_format_operational_error_includes_errno() -> None:
    class FakeOrig(Exception):
        args = (1045, "Access denied for user 'admin'@'10.0.0.1'")

    exc = OperationalError("statement", {}, FakeOrig())
    assert "1045" in format_operational_error(exc)


def test_database_startup_error_message_includes_target_and_mysql_detail() -> None:
    class FakeOrig(Exception):
        args = (1045, "Access denied")

    exc = OperationalError("statement", {}, FakeOrig())
    message = database_startup_error_message(exc, "user=admin host=example.com database=db")
    assert "user=admin" in message
    assert "1045" in message
    assert "Add User To Database" in message


def test_operational_error_tips_1045() -> None:
    class FakeOrig(Exception):
        args = (1045, "Access denied")

    exc = OperationalError("statement", {}, FakeOrig())
    assert "1045" in operational_error_tips(exc)
    assert "Add User To Database" in operational_error_tips(exc)


def test_operational_error_tips_2003() -> None:
    class FakeOrig(Exception):
        args = (2003, "Can't connect")

    exc = OperationalError("statement", {}, FakeOrig())
    tips = operational_error_tips(exc)
    assert "2003" in tips
    assert "Remote MySQL" in tips
