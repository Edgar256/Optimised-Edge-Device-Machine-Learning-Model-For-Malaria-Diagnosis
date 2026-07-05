"""Tests for authentication and patient record APIs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _signup_payload(**overrides) -> dict:
    payload = {
        "first_name": "Jane",
        "last_name": "Nurse",
        "email": "jane@example.com",
        "phone": "+256700000001",
        "job_title": "Clinical Officer",
        "health_facility_name": "Kasomoro health centre",
        "password": "securepass123",
    }
    payload.update(overrides)
    return payload


def test_signup_login_and_me(client: TestClient) -> None:
    signup = client.post("/v1/auth/signup", json=_signup_payload())
    assert signup.status_code == 201
    signup_body = signup.json()
    assert signup_body["token_type"] == "bearer"
    assert signup_body["user"]["email"] == "jane@example.com"
    assert signup_body["user"]["user_type"] == "MEDICAL_PERSONNEL"
    token = signup_body["access_token"]

    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["job_title"] == "Clinical Officer"

    login = client.post(
        "/v1/auth/login",
        json={"email": "jane@example.com", "password": "securepass123", "remember_me": True},
    )
    assert login.status_code == 200
    assert login.json()["remember_me"] is True
    assert login.json()["expires_in"] == 30 * 24 * 60 * 60


def test_signup_duplicate_email_returns_409(client: TestClient) -> None:
    client.post("/v1/auth/signup", json=_signup_payload())
    duplicate = client.post("/v1/auth/signup", json=_signup_payload())
    assert duplicate.status_code == 409


def test_login_invalid_password_returns_401(client: TestClient) -> None:
    client.post("/v1/auth/signup", json=_signup_payload())
    login = client.post(
        "/v1/auth/login",
        json={"email": "jane@example.com", "password": "wrong-password"},
    )
    assert login.status_code == 401


def test_patient_crud_with_csv_fields(client: TestClient) -> None:
    signup = client.post("/v1/auth/signup", json=_signup_payload(email="records@example.com"))
    token = signup.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    create = client.post(
        "/v1/patients",
        headers=headers,
        json={
            "patient_id_anonymous_code": "Kasomoro001",
            "health_facility_name": "Kasomoro health centre 2",
            "date_of_visit": "2026-06-25",
            "age": 10,
            "gender": "female",
            "geographical_zone": "Kanyamunyu",
            "fever": "yes",
            "headache": "yes",
            "chills": "yes",
            "fatigue": "no",
            "fever_duration_days": 3,
            "anemia_signs": "no",
            "season_of_visit": "Dry",
            "recent_travel": "no",
            "exposure_risk": "yes",
            "household_malaria_history": "yes",
            "rdt_result": "positive",
            "microscopy_result": "",
            "final_confirmed_diagnosis": "Malaria",
            "treatment_given": "Coartem",
            "latitude": 1.4332,
            "longitude": 31.3521,
        },
    )
    assert create.status_code == 201
    created = create.json()
    assert created["patient_id_anonymous_code"] == "Kasomoro001"
    assert created["latitude"] == pytest.approx(1.4332)
    patient_id = created["id"]

    listed = client.get("/v1/patients", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    updated = client.put(
        f"/v1/patients/{patient_id}",
        headers=headers,
        json={"additional_clinical_notes": "Follow-up in 3 days"},
    )
    assert updated.status_code == 200
    assert updated.json()["additional_clinical_notes"] == "Follow-up in 3 days"

    deleted = client.delete(f"/v1/patients/{patient_id}", headers=headers)
    assert deleted.status_code == 204
    assert client.get("/v1/patients", headers=headers).json() == []


def test_patients_require_authentication(client: TestClient) -> None:
    response = client.get("/v1/patients")
    assert response.status_code == 401


def _admin_signup_payload(**overrides) -> dict:
    payload = {
        "first_name": "Alex",
        "last_name": "Admin",
        "email": "admin@example.com",
        "phone": "+256700000099",
        "job_title": "System Administrator",
        "password": "securepass123",
        "admin_registration_secret": "test-admin-secret",
    }
    payload.update(overrides)
    return payload


def test_admin_register_and_login(client: TestClient) -> None:
    register = client.post("/v1/auth/admin/register", json=_admin_signup_payload())
    assert register.status_code == 201
    body = register.json()
    assert body["user"]["user_type"] == "ADMIN"
    token = body["access_token"]

    admin_me = client.get("/v1/auth/admin/me", headers={"Authorization": f"Bearer {token}"})
    assert admin_me.status_code == 200

    login = client.post(
        "/v1/auth/admin/login",
        json={"email": "admin@example.com", "password": "securepass123", "remember_me": True},
    )
    assert login.status_code == 200
    assert login.json()["user"]["user_type"] == "ADMIN"


def test_admin_cannot_use_mobile_login(client: TestClient) -> None:
    client.post("/v1/auth/admin/register", json=_admin_signup_payload())
    login = client.post(
        "/v1/auth/login",
        json={"email": "admin@example.com", "password": "securepass123"},
    )
    assert login.status_code == 403


def test_medical_personnel_cannot_use_admin_login(client: TestClient) -> None:
    client.post("/v1/auth/signup", json=_signup_payload())
    login = client.post(
        "/v1/auth/admin/login",
        json={"email": "jane@example.com", "password": "securepass123"},
    )
    assert login.status_code == 403


def test_admin_register_rejects_invalid_secret(client: TestClient) -> None:
    response = client.post(
        "/v1/auth/admin/register",
        json=_admin_signup_payload(admin_registration_secret="wrong-secret"),
    )
    assert response.status_code == 403
