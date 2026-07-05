"""Tests for serving the built React dashboard from FastAPI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.deployment.api import _frontend_dist_dir, create_app


def test_serves_dashboard_when_dist_exists() -> None:
    if not (_frontend_dist_dir() / "index.html").is_file():
        return

    with TestClient(create_app()) as client:
        response = client.get("/login")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Malaria Edge ML" in response.text

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
