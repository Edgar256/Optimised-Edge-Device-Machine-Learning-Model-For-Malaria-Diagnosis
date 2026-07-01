"""Pydantic schemas for the offline prediction API."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Gender(str, Enum):
    male = "male"
    female = "female"


class DiagnosticResult(str, Enum):
    positive = "positive"
    negative = "negative"
    not_performed = "not_performed"


class SymptomsInput(BaseModel):
    """Clinical symptoms aligned with the Kobo intake form."""

    model_config = ConfigDict(extra="forbid")

    fever: bool = False
    headache: bool = False
    chills: bool = False
    vomiting: bool = False
    fatigue: bool = False
    anemia_signs: bool = False
    fever_duration_days: float | None = Field(
        default=None,
        ge=0,
        le=60,
        description="Days since fever onset; imputed from training data when omitted.",
    )
    other: str | None = Field(
        default=None,
        max_length=200,
        description="Free-text other symptoms (mapped to Other Symptoms field).",
    )


class PredictionRequest(BaseModel):
    """Patient payload for offline malaria risk scoring."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 12,
                "gender": "female",
                "temperature_celsius": 38.5,
                "symptoms": {
                    "fever": True,
                    "headache": True,
                    "chills": True,
                    "vomiting": False,
                    "fatigue": True,
                    "anemia_signs": False,
                    "fever_duration_days": 3,
                    "other": None,
                },
                "rdt": "not_performed",
                "microscopy": "not_performed",
            }
        },
    )

    age: float = Field(..., ge=0, le=120)
    gender: Gender
    temperature_celsius: float | None = Field(
        default=None,
        ge=30,
        le=45,
        description="Body temperature in Celsius; used for clinical risk escalation only.",
    )
    symptoms: SymptomsInput = Field(default_factory=SymptomsInput)
    rdt: DiagnosticResult = DiagnosticResult.not_performed
    microscopy: DiagnosticResult = DiagnosticResult.not_performed

    health_facility_name: str | None = Field(
        default=None,
        description="Optional facility name; defaults to project training default.",
    )
    geographical_zone: str | None = None
    season: Literal["Dry", "Rainy"] | None = None
    recent_travel: bool | None = None
    exposure_risk: bool | None = None
    household_malaria_history: bool | None = None


class PredictionResponse(BaseModel):
    """Structured prediction for Android / edge clients."""

    model_config = ConfigDict(extra="forbid")

    prediction: Literal["Malaria", "Not malaria"]
    confidence_score: float = Field(..., ge=0, le=1)
    risk_category: Literal["low", "medium", "high"]
    malaria_probability: float = Field(..., ge=0, le=1)
    model_name: str
    api_version: str = "v1"
    clinical_flags: list[str] = Field(
        default_factory=list,
        description="Human-readable flags explaining risk escalation (RDT, fever, temperature).",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    preprocessing_loaded: bool
    offline: bool = True
    api_version: str = "v1"
