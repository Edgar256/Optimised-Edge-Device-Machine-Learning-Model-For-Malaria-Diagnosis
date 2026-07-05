"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from src.database.session import reset_engine
from src.deployment.api import create_app

# Use an isolated in-memory database during tests unless the runner already set DATABASE_URL.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("ADMIN_REGISTRATION_SECRET", "test-admin-secret")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ADMIN_REGISTRATION_SECRET", "test-admin-secret")
    reset_engine()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    reset_engine()
