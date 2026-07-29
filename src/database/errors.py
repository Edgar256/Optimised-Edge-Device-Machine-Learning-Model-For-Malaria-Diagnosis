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
            "MySQL rejected the username or password (1045). "
            "(1) Copy the exact password from your provider console (Aiven: Service → Users → reset/copy). "
            "(2) Confirm DATABASE_URL user, password, host, port, and database name match the console. "
            "(3) On Aiven, open Connection info → Allowed IP addresses and allow Render "
            "(0.0.0.0/0 for public access, or add the IP shown in the error). "
            "(4) URL-encode special characters in the password (@ → %40, # → %23)."
        )
    if errno == 2003:
        return (
            "Cannot reach the MySQL host (2003). Check host/port, allow public/network access "
            "in the provider console (Aiven Allowed IP addresses), and ensure SSL is enabled "
            "(?ssl-mode=REQUIRED or ?ssl=true) if the provider requires it."
        )
    if errno == 1049:
        return (
            "Database does not exist (1049). Create the database in the provider console "
            "(Aiven: Databases) and use that exact name in the DATABASE_URL path."
        )
    return (
        "Confirm host/port/user/password/database in DATABASE_URL, allow the client IP "
        "in the provider firewall, and use ?ssl-mode=REQUIRED or ?ssl=true when SSL is required."
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
