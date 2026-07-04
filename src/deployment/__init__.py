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
from src.deployment.tflite_export import TfliteExportResult, bump_semver, export_tflite_model

__all__ = [
    "DiagnosticResult",
    "Gender",
    "HealthResponse",
    "PredictorArtifacts",
    "PredictionRequest",
    "PredictionResponse",
    "SymptomsInput",
    "TfliteExportResult",
    "app",
    "build_inference_dataframe",
    "bump_semver",
    "create_app",
    "export_tflite_model",
    "load_predictor_artifacts",
    "predict_malaria",
    "transform_for_inference",
]
