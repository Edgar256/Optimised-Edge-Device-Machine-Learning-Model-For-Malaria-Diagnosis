"""Tests for the offline prediction API."""

import pytest
from fastapi.testclient import TestClient

from src.deployment.api import create_app
from src.deployment.predictor import (
    build_inference_dataframe,
    classify_risk_category,
    load_predictor_artifacts,
    predict_malaria,
)
from src.deployment.schemas import DiagnosticResult, Gender, PredictionRequest, SymptomsInput


def _sample_request(**overrides) -> PredictionRequest:
    payload = {
        "age": 8,
        "gender": Gender.female,
        "temperature_celsius": 38.2,
        "symptoms": SymptomsInput(
            fever=True,
            headache=True,
            chills=True,
            vomiting=False,
            fatigue=True,
            fever_duration_days=2,
        ),
        "rdt": DiagnosticResult.not_performed,
        "microscopy": DiagnosticResult.not_performed,
    }
    payload.update(overrides)
    return PredictionRequest(**payload)


def test_build_inference_dataframe_maps_symptoms() -> None:
    request = _sample_request()
    df = build_inference_dataframe(request)
    assert df.loc[0, "Age"] == 8
    assert df.loc[0, "Fever (Yes/No)"] == "yes"
    assert df.loc[0, "Vomiting (Yes/No)"] == "no"


def test_classify_risk_category_rdt_positive() -> None:
    request = _sample_request(rdt=DiagnosticResult.positive)
    risk, flags = classify_risk_category(0.1, request)
    assert risk == "high"
    assert any("RDT" in flag for flag in flags)


@pytest.fixture(scope="module")
def artifacts():
    try:
        return load_predictor_artifacts()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_predict_malaria_returns_response(artifacts) -> None:
    response = predict_malaria(_sample_request(), artifacts)
    assert response.prediction in {"Malaria", "Not malaria"}
    assert 0.0 <= response.confidence_score <= 1.0
    assert response.risk_category in {"low", "medium", "high"}
    assert response.model_name == artifacts.model_name


def test_api_health_and_predict(artifacts) -> None:
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["offline"] is True
        assert body["model_loaded"] is True

        predict_response = client.post("/v1/predict", json=_sample_request().model_dump())
        assert predict_response.status_code == 200
        result = predict_response.json()
        assert "prediction" in result
        assert "confidence_score" in result
        assert "risk_category" in result
