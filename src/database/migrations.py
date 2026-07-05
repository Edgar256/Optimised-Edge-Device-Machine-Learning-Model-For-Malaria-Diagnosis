"""Lightweight schema upgrades for existing databases."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from src.database.enums import UserType


def _column_names(engine: Engine, table: str) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def _add_column(engine: Engine, table: str, ddl: str) -> None:
    with engine.begin() as connection:
        connection.execute(text(ddl))


def migrate_schema(engine: Engine) -> list[str]:
    """Apply additive migrations. Returns human-readable actions taken."""
    applied: list[str] = []
    default_user_type = UserType.MEDICAL_PERSONNEL.value

    user_columns = _column_names(engine, "users")
    if user_columns and "user_type" not in user_columns:
        if engine.dialect.name == "mysql":
            ddl = (
                f"ALTER TABLE users ADD COLUMN user_type VARCHAR(32) "
                f"NOT NULL DEFAULT '{default_user_type}'"
            )
        else:
            ddl = (
                f"ALTER TABLE users ADD COLUMN user_type VARCHAR(32) "
                f"NOT NULL DEFAULT '{default_user_type}'"
            )
        _add_column(engine, "users", ddl)
        applied.append("Added users.user_type column")

    patient_columns = _column_names(engine, "patients")
    if patient_columns and "latitude" not in patient_columns:
        _add_column(engine, "patients", "ALTER TABLE patients ADD COLUMN latitude FLOAT NULL")
        applied.append("Added patients.latitude column")
    if patient_columns and "longitude" not in patient_columns:
        _add_column(engine, "patients", "ALTER TABLE patients ADD COLUMN longitude FLOAT NULL")
        applied.append("Added patients.longitude column")

    return applied
