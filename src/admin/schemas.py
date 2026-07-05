"""Admin dashboard API schemas."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from src.database.enums import UserType
from src.patients.schemas import PatientResponse


class DashboardStatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_patients: int
    malaria_confirmed: int
    not_malaria: int
    pending_diagnosis: int
    total_medical_personnel: int
    total_admins: int
    model_loaded: bool
    model_name: str | None


class AdminUserSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    first_name: str
    last_name: str
    email: EmailStr
    phone: str
    job_title: str
    health_facility_name: str | None
    user_type: UserType
    created_at: datetime


class AdminPatientResponse(PatientResponse):
    created_by_name: str
    created_by_email: EmailStr
    created_by_facility: str | None
