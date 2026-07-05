"""Patient record Pydantic schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["yes", "no", ""]
Season = Literal["Dry", "Rainy", ""]
Diagnosis = Literal["Malaria", "Not malaria", ""]
DiagnosticResult = Literal["positive", "negative", ""]


class PatientBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime | None = None
    end: datetime | None = None
    patient_id_anonymous_code: str | None = Field(default=None, max_length=120)
    health_facility_name: str | None = Field(default=None, max_length=255)
    date_of_visit: date | None = None
    age: float | None = Field(default=None, ge=0, le=120)
    gender: str | None = Field(default=None, max_length=20)
    geographical_zone: str | None = Field(default=None, max_length=255)

    fever: YesNo | None = None
    headache: YesNo | None = None
    chills: YesNo | None = None
    vomiting: YesNo | None = None
    fatigue: YesNo | None = None
    other_symptoms_specify: str | None = None
    fever_duration_days: float | None = Field(default=None, ge=0, le=60)
    anemia_signs: YesNo | None = None
    season_of_visit: Season | None = None
    recent_travel: YesNo | None = None
    exposure_risk: YesNo | None = None
    household_malaria_history: YesNo | None = None

    rdt_result: DiagnosticResult | None = None
    microscopy_result: DiagnosticResult | None = None
    parasite_density: str | None = Field(default=None, max_length=120)
    final_confirmed_diagnosis: Diagnosis | None = None
    treatment_given: str | None = None
    additional_clinical_notes: str | None = None

    kobo_id: str | None = Field(default=None, max_length=64)
    kobo_uuid: str | None = Field(default=None, max_length=64)
    submission_time: datetime | None = None
    validation_status: str | None = Field(default=None, max_length=120)
    notes: str | None = None
    status: str | None = Field(default=None, max_length=80)
    submitted_by: str | None = Field(default=None, max_length=120)
    form_version: str | None = Field(default=None, max_length=80)
    tags: str | None = Field(default=None, max_length=255)
    meta_root_uuid: str | None = Field(default=None, max_length=80)
    kobo_index: int | None = None

    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class PatientCreateRequest(PatientBase):
    pass


class PatientUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime | None = None
    end: datetime | None = None
    patient_id_anonymous_code: str | None = Field(default=None, max_length=120)
    health_facility_name: str | None = Field(default=None, max_length=255)
    date_of_visit: date | None = None
    age: float | None = Field(default=None, ge=0, le=120)
    gender: str | None = Field(default=None, max_length=20)
    geographical_zone: str | None = Field(default=None, max_length=255)
    fever: YesNo | None = None
    headache: YesNo | None = None
    chills: YesNo | None = None
    vomiting: YesNo | None = None
    fatigue: YesNo | None = None
    other_symptoms_specify: str | None = None
    fever_duration_days: float | None = Field(default=None, ge=0, le=60)
    anemia_signs: YesNo | None = None
    season_of_visit: Season | None = None
    recent_travel: YesNo | None = None
    exposure_risk: YesNo | None = None
    household_malaria_history: YesNo | None = None
    rdt_result: DiagnosticResult | None = None
    microscopy_result: DiagnosticResult | None = None
    parasite_density: str | None = Field(default=None, max_length=120)
    final_confirmed_diagnosis: Diagnosis | None = None
    treatment_given: str | None = None
    additional_clinical_notes: str | None = None
    kobo_id: str | None = Field(default=None, max_length=64)
    kobo_uuid: str | None = Field(default=None, max_length=64)
    submission_time: datetime | None = None
    validation_status: str | None = Field(default=None, max_length=120)
    notes: str | None = None
    status: str | None = Field(default=None, max_length=80)
    submitted_by: str | None = Field(default=None, max_length=120)
    form_version: str | None = Field(default=None, max_length=80)
    tags: str | None = Field(default=None, max_length=255)
    meta_root_uuid: str | None = Field(default=None, max_length=80)
    kobo_index: int | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class PatientResponse(PatientBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_by_user_id: int
    created_at: datetime
    updated_at: datetime
