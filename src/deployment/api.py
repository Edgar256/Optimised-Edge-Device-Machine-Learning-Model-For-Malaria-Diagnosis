"""FastAPI application for offline malaria prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.deployment.predictor import PredictorArtifacts, load_predictor_artifacts, predict_malaria
from src.deployment.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.utils.config import load_config

API_DESCRIPTION = """
Offline malaria triage API for edge devices and Android clients.

- **No cloud dependency** — model and preprocessing load from local `models/` and `data/processed/`.
- **Android integration** — POST JSON to `/v1/predict`; OpenAPI schema at `/openapi.json`.
- **LAN hosting** — bind to `0.0.0.0` and call `http://<device-ip>:8000/v1/predict` from the app.

The ML model uses triage-time clinical features. Temperature, RDT, and microscopy are accepted
for bedside workflow and **risk_category** escalation; RDT/microscopy are excluded from model
features to avoid label leakage at intake time.
"""

_artifacts: PredictorArtifacts | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _artifacts
    _artifacts = load_predictor_artifacts()
    yield
    _artifacts = None


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Build the FastAPI application (used by uvicorn and tests)."""
    cfg = config or load_config()
    api_cfg = cfg.get("api", {})

    app = FastAPI(
        title="Malaria Edge ML — Offline Prediction API",
        description=API_DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    cors_origins = api_cfg.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        loaded = _artifacts is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_name=_artifacts.model_name if loaded else None,
            preprocessing_loaded=loaded,
        )

    @app.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "service": "malaria-edge-ml",
            "mode": "offline",
            "predict": "/v1/predict",
            "health": "/health",
            "openapi": "/openapi.json",
        }

    @app.post(
        "/v1/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
        summary="Predict malaria diagnosis from clinical intake",
    )
    def predict(request_body: PredictionRequest) -> PredictionResponse:
        if _artifacts is None:
            raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
        try:
            return predict_malaria(request_body, _artifacts, config=cfg)
        except Exception as exc:  # noqa: BLE001 — surface inference errors to client
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return app


app = create_app()
