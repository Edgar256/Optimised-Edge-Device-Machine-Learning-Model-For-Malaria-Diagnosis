"""Tests for additive database migrations."""

from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, insert, inspect

from src.database.migrations import migrate_schema


def test_migrate_schema_adds_user_type_column() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    users = Table(
        "users",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("email", String(255), nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(insert(users).values(id=1, email="legacy@example.com"))

    applied = migrate_schema(engine)
    assert "Added users.user_type column" in applied

    columns = {col["name"] for col in inspect(engine).get_columns("users")}
    assert "user_type" in columns
