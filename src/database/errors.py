"""Database connection error helpers."""

from __future__ import annotations

from sqlalchemy.exc import OperationalError


def format_operational_error(exc: OperationalError) -> str:
    """Return a log-safe summary of a SQLAlchemy OperationalError."""
    orig = exc.orig
    if orig is not None and getattr(orig, "args", None):
        return str(orig.args[0]) if len(orig.args) == 1 else str(orig.args)
    return str(exc)


def operational_error_errno(exc: OperationalError) -> int | None:
    """Return the MySQL errno from an OperationalError, if present."""
    orig = exc.orig
    if orig is not None and getattr(orig, "args", None) and orig.args:
        first = orig.args[0]
        if isinstance(first, int):
            return first
    return None


def operational_error_tips(exc: OperationalError) -> str:
    """Return actionable tips for common MySQL connection errors."""
    errno = operational_error_errno(exc)
    if errno == 1045:
        return (
            "MySQL rejected the username or password (1045). In Cloudsters/cPanel: "
            "(1) reset the MySQL password for this user, "
            "(2) use Databases → Add User To Database to grant ALL PRIVILEGES on the database, "
            "(3) confirm DATABASE_URL uses that exact user, password, and database name."
        )
    if errno == 2003:
        return (
            "Cannot reach the MySQL host (2003). Open port 3306 on Cloudsters, add your IP "
            "under Databases → Remote MySQL (or use % for any host). "
            "If Render gets 1045 but local check-db gets 2003, your home IP may be blocked — "
            "fix credentials in Cloudsters first, then redeploy Render."
        )
    if errno == 1049:
        return "Database does not exist (1049). Create it in Cloudsters and use the exact name in DATABASE_URL."
    return (
        "Confirm Cloudsters Remote MySQL allows % (any host), user is linked to the database, "
        "and try appending ?ssl=true to DATABASE_URL if the provider requires SSL."
    )


def database_startup_error_message(exc: OperationalError, target: str) -> str:
    """Build a startup failure message for MySQL connection errors."""
    mysql_detail = format_operational_error(exc)
    tips = operational_error_tips(exc)
    return (
        "Database connection failed during startup. "
        f"Target: {target}. MySQL error: {mysql_detail}. "
        f"{tips} "
        "Run `python main.py check-db` with the same DATABASE_URL when your network can reach the host."
    )
