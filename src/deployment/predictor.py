"""Offline malaria prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.deployment.schemas import DiagnosticResult, Gender, PredictionRequest, PredictionResponse
from src.evaluation.explainability import load_model_artifact, resolve_best_model_name
from src.preprocessing.common import model_feature_columns
from src.utils.config import load_config
from src.utils.paths import resolve_path

INFERENCE_PIPELINE_STEPS = (
    "category_normalizer",
    "impossible_values",
    "date_features",
    "feature_selector",
    "column_transformer",
)

YES_NO_COLUMN_MAP = {
    "fever": "Fever (Yes/No)",
    "headache": "Headache (Yes/No)",
    "chills": "Chills (Yes/No)",
    "vomiting": "Vomiting (Yes/No)",
    "fatigue": "Fatigue (Yes/No)",
    "anemia_signs": "Anemia Signs (Yes/No)",
}

POST_DIAGNOSIS_COLUMN_MAP = {
    "rdt": "Rapid Diagnostic Test (RDT) Result (Positive/Negative)",
    "microscopy": "Microscopy Result (Positive/Negative)",
}


@dataclass
class PredictorArtifacts:
    model_name: str
    estimator: Any
    preprocessing_pipeline: Pipeline
    feature_names: list[str]


def _api_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("api", {})


def _bool_to_yes_no(value: bool | None) -> str | float:
    if value is None:
        return np.nan
    return "yes" if value else "no"


def _diagnostic_to_form_value(result: DiagnosticResult) -> str | float:
    if result == DiagnosticResult.positive:
        return "Positive"
    if result == DiagnosticResult.negative:
        return "Negative"
    return np.nan


def build_inference_dataframe(
    request: PredictionRequest, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Map an API request to a single-row DataFrame compatible with the preprocessing pipeline."""
    cfg = config or load_config()
    api_cfg = _api_config(cfg)
    defaults = api_cfg.get("defaults", {})
    data_cfg = cfg["data"]
    today = date.today().isoformat()

    row: dict[str, Any] = {column: np.nan for column in model_feature_columns()}
    row.update(
        {
            "Age": request.age,
            "Gender (Male/Female)": request.gender.value,
            "Health Facility Name": request.health_facility_name
            or defaults.get("health_facility_name", "kasomoro_health_centre"),
            "Geographical Zone": request.geographical_zone
            or defaults.get("geographical_zone", "mbaraara"),
            "Date of Visit": today,
            "Season of Visit (Dry/Rainy)": request.season or defaults.get("season", "Rainy"),
            "Recent Travel (Yes/No)": _bool_to_yes_no(request.recent_travel),
            "Exposure Risk (e.g., mosquito-prone area) (Yes/No)": _bool_to_yes_no(
                request.exposure_risk
            ),
            "Household Malaria History (Yes/No)": _bool_to_yes_no(
                request.household_malaria_history
            ),
            "Fever Duration (Days)": request.symptoms.fever_duration_days,
            "Other Symptoms (Specify)": request.symptoms.other,
            data_cfg["target_column"]: data_cfg["target_negative_label"],
            "Patient ID (Anonymous Code)": "api_inference",
            POST_DIAGNOSIS_COLUMN_MAP["rdt"]: _diagnostic_to_form_value(request.rdt),
            POST_DIAGNOSIS_COLUMN_MAP["microscopy"]: _diagnostic_to_form_value(request.microscopy),
        }
    )

    for symptom_field, column_name in YES_NO_COLUMN_MAP.items():
        row[column_name] = _bool_to_yes_no(getattr(request.symptoms, symptom_field))

    return pd.DataFrame([row])


def transform_for_inference(df: pd.DataFrame, pipeline: Pipeline) -> np.ndarray:
    """Run inference-safe preprocessing steps (skips training-only row filtering)."""
    transformed = df.copy()
    for step_name in INFERENCE_PIPELINE_STEPS:
        transformed = pipeline.named_steps[step_name].transform(transformed)
    return np.asarray(transformed)


def _risk_thresholds(config: dict[str, Any]) -> dict[str, float]:
    api_cfg = _api_config(config)
    thresholds = api_cfg.get("risk_thresholds", {})
    return {
        "high": float(thresholds.get("high_confidence", 0.75)),
        "medium": float(thresholds.get("medium_confidence", 0.5)),
        "fever_temperature_celsius": float(thresholds.get("fever_temperature_celsius", 38.0)),
    }


def classify_risk_category(
    malaria_probability: float,
    request: PredictionRequest,
    config: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Assign clinical risk tier using model score plus optional bedside signals."""
    cfg = config or load_config()
    thresholds = _risk_thresholds(cfg)
    flags: list[str] = []

    if request.rdt == DiagnosticResult.positive:
        flags.append("RDT positive — high clinical risk")
        return "high", flags
    if request.microscopy == DiagnosticResult.positive:
        flags.append("Microscopy positive — high clinical risk")
        return "high", flags

    if malaria_probability >= thresholds["high"]:
        flags.append("Model malaria probability above high threshold")
        return "high", flags

    if malaria_probability >= thresholds["medium"]:
        flags.append("Model malaria probability in medium range")
        return "medium", flags

    elevated_temp = (
        request.temperature_celsius is not None
        and request.temperature_celsius >= thresholds["fever_temperature_celsius"]
    )
    if elevated_temp and request.symptoms.fever:
        flags.append("Elevated temperature with reported fever")
        return "medium", flags

    return "low", flags


def predict_malaria(
    request: PredictionRequest,
    artifacts: PredictorArtifacts,
    config: dict[str, Any] | None = None,
) -> PredictionResponse:
    """Score a single patient request offline."""
    cfg = config or load_config()
    data_cfg = cfg["data"]

    inference_df = build_inference_dataframe(request, config=cfg)
    features = transform_for_inference(inference_df, artifacts.preprocessing_pipeline)

    if hasattr(artifacts.estimator, "predict_proba"):
        probabilities = artifacts.estimator.predict_proba(features)[0]
        classes = list(artifacts.estimator.classes_)
        malaria_index = classes.index(1) if 1 in classes else int(np.argmax(probabilities))
        malaria_probability = float(probabilities[malaria_index])
    else:
        prediction_value = int(artifacts.estimator.predict(features)[0])
        malaria_probability = float(prediction_value)

    positive_label = data_cfg["target_positive_label"]
    negative_label = data_cfg["target_negative_label"]
    prediction_label = positive_label if malaria_probability >= 0.5 else negative_label
    confidence_score = (
        malaria_probability if prediction_label == positive_label else 1.0 - malaria_probability
    )

    risk_category, clinical_flags = classify_risk_category(malaria_probability, request, config=cfg)

    return PredictionResponse(
        prediction=prediction_label,  # type: ignore[arg-type]
        confidence_score=round(confidence_score, 4),
        risk_category=risk_category,  # type: ignore[arg-type]
        malaria_probability=round(malaria_probability, 4),
        model_name=artifacts.model_name,
        clinical_flags=clinical_flags,
    )


def load_predictor_artifacts(config: dict[str, Any] | None = None) -> PredictorArtifacts:
    """Load the best model and fitted preprocessing pipeline from local disk."""
    cfg = config or load_config()
    api_cfg = _api_config(cfg)
    model_name = str(api_cfg.get("model_name") or resolve_best_model_name(cfg))

    artifact = load_model_artifact(model_name, config=cfg)
    pipeline_rel = artifact.get(
        "preprocessing_pipeline_path", "data/processed/preprocessing_pipeline.joblib"
    )
    pipeline_path = resolve_path(pipeline_rel)
    if not pipeline_path.is_file():
        pipeline_path = resolve_path(
            Path(cfg["paths"]["processed_data_dir"]) / "preprocessing_pipeline.joblib"
        )
    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at {pipeline_path}. Run `python main.py preprocess` first."
        )

    preprocessing_pipeline = joblib.load(pipeline_path)
    feature_names = (
        preprocessing_pipeline.named_steps["column_transformer"].get_feature_names_out().tolist()
    )

    return PredictorArtifacts(
        model_name=model_name,
        estimator=artifact["estimator"],
        preprocessing_pipeline=preprocessing_pipeline,
        feature_names=feature_names,
    )
