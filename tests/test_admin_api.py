"""Tests for admin dashboard API."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _admin_signup_payload(**overrides) -> dict:
    payload = {
        "first_name": "Alex",
        "last_name": "Admin",
        "email": "admin-stats@example.com",
        "phone": "+256700000099",
        "job_title": "System Administrator",
        "password": "securepass123",
        "admin_registration_secret": "test-admin-secret",
    }
    payload.update(overrides)
    return payload


def test_admin_stats_and_patients(client: TestClient) -> None:
    admin = client.post("/v1/auth/admin/register", json=_admin_signup_payload())
    assert admin.status_code == 201
    token = admin.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    stats = client.get("/v1/admin/stats", headers=headers)
    assert stats.status_code == 200
    body = stats.json()
    assert "total_patients" in body
    assert body["model_loaded"] is True

    users = client.get("/v1/admin/users", headers=headers)
    assert users.status_code == 200
    assert any(u["user_type"] == "ADMIN" for u in users.json())

    patients = client.get("/v1/admin/patients", headers=headers)
    assert patients.status_code == 200
    assert isinstance(patients.json(), list)


def test_admin_endpoints_require_admin_token(client: TestClient) -> None:
    medical = client.post(
        "/v1/auth/signup",
        json={
            "first_name": "Jane",
            "last_name": "Nurse",
            "email": "nurse@example.com",
            "phone": "+256700000001",
            "job_title": "Clinical Officer",
            "password": "securepass123",
        },
    )
    token = medical.json()["access_token"]
    response = client.get("/v1/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
