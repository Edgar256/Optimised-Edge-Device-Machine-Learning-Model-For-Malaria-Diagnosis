"""Edge inference packaging, export, and runtime integration."""

from src.deployment.api import app, create_app
from src.deployment.predictor import (
    PredictorArtifacts,
    build_inference_dataframe,
    load_predictor_artifacts,
    predict_malaria,
    transform_for_inference,
)
from src.deployment.schemas import (
    DiagnosticResult,
    Gender,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    SymptomsInput,
)

__all__ = [
    "DiagnosticResult",
    "Gender",
    "HealthResponse",
    "PredictorArtifacts",
    "PredictionRequest",
    "PredictionResponse",
    "SymptomsInput",
    "app",
    "build_inference_dataframe",
    "create_app",
    "load_predictor_artifacts",
    "predict_malaria",
    "transform_for_inference",
]
