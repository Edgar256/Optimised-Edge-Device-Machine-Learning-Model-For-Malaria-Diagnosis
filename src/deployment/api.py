"""FastAPI application for offline malaria prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.admin.routes import router as admin_router
from src.auth.routes import router as auth_router
from src.deployment.predictor import load_predictor_artifacts, predict_malaria
from src.deployment.runtime import get_predictor_artifacts, set_predictor_artifacts
from src.deployment.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.patients.routes import router as patients_router
from src.utils.config import load_config
from src.utils.env import get_database_url
from src.utils.paths import find_project_root

API_DESCRIPTION = """
Offline malaria triage API for edge devices and Android clients.

- **No cloud dependency** — model and preprocessing load from local `models/` and `data/processed/`.
- **Android integration** — POST JSON to `/v1/predict`; OpenAPI schema at `/openapi.json`.
- **LAN hosting** — bind to `0.0.0.0` and call `http://<device-ip>:8000/v1/predict` from the app.
- **Accounts & records** — signup/login at `/v1/auth/*`; patient visits at `/v1/patients`.

The ML model uses triage-time clinical features. Temperature, RDT, and microscopy are accepted
for bedside workflow and **risk_category** escalation; RDT/microscopy are excluded from model
features to avoid label leakage at intake time.
"""


def _frontend_dist_dir() -> Path:
    return find_project_root() / "frontend" / "dist"


def _mount_frontend_dashboard(app: FastAPI) -> bool:
    """Serve the built React admin dashboard when frontend/dist exists."""
    dist_dir = _frontend_dist_dir()
    index_file = dist_dir / "index.html"
    if not index_file.is_file():
        return False

    @app.get("/", include_in_schema=False)
    def serve_dashboard_root() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_dashboard_path(full_path: str) -> FileResponse:
        if full_path.startswith("v1/"):
            raise HTTPException(status_code=404, detail="Not found.")
        asset = dist_dir / full_path
        if full_path and asset.is_file():
            return FileResponse(asset)
        return FileResponse(index_file)

    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        set_predictor_artifacts(load_predictor_artifacts())
    except FileNotFoundError as exc:
        root = find_project_root()
        model_path = root / "models" / "optimized" / "logistic_regression.joblib"
        pipeline_path = root / "data" / "processed" / "preprocessing_pipeline.joblib"
        raise RuntimeError(
            f"{exc} Deploy requires committed artifacts at {model_path} "
            f"(exists={model_path.is_file()}) and {pipeline_path} "
            f"(exists={pipeline_path.is_file()}). Project root={root}."
        ) from exc
    if get_database_url():
        from src.database.session import init_db

        init_db()
    yield
    set_predictor_artifacts(None)


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
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(auth_router)
    app.include_router(patients_router)
    app.include_router(admin_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        artifacts = get_predictor_artifacts()
        loaded = artifacts is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_name=artifacts.model_name if loaded else None,
            preprocessing_loaded=loaded,
        )

    if not _mount_frontend_dashboard(app):

        @app.get("/", tags=["system"])
        def root() -> dict[str, str]:
            return {
                "service": "malaria-edge-ml",
                "mode": "offline",
                "predict": "/v1/predict",
                "auth_signup": "/v1/auth/signup",
                "auth_login": "/v1/auth/login",
                "auth_admin_register": "/v1/auth/admin/register",
                "auth_admin_login": "/v1/auth/admin/login",
                "patients": "/v1/patients",
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
        artifacts = get_predictor_artifacts()
        if artifacts is None:
            raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
        try:
            return predict_malaria(request_body, artifacts, config=cfg)
        except Exception as exc:  # noqa: BLE001 — surface inference errors to client
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return app


app = create_app()
