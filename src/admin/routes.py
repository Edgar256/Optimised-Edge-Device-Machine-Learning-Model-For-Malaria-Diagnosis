"""Admin dashboard API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.admin.schemas import AdminPatientResponse, AdminUserSummary, DashboardStatsResponse
from src.auth.dependencies import get_current_admin
from src.database.enums import UserType
from src.database.models import Patient, User
from src.database.session import get_db
from src.deployment.runtime import get_predictor_artifacts
from src.patients.schemas import PatientResponse

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _patient_to_admin_response(patient: Patient) -> AdminPatientResponse:
    creator = patient.created_by_user
    base = PatientResponse.model_validate(patient)
    return AdminPatientResponse(
        **base.model_dump(),
        created_by_name=f"{creator.first_name} {creator.last_name}",
        created_by_email=creator.email,
        created_by_facility=creator.health_facility_name,
    )


@router.get("/stats", response_model=DashboardStatsResponse)
def dashboard_stats(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> DashboardStatsResponse:
    total_patients = db.query(func.count(Patient.id)).scalar() or 0
    malaria_confirmed = (
        db.query(func.count(Patient.id))
        .filter(Patient.final_confirmed_diagnosis.ilike("malaria"))
        .filter(~Patient.final_confirmed_diagnosis.ilike("not%"))
        .scalar()
        or 0
    )
    not_malaria = (
        db.query(func.count(Patient.id))
        .filter(Patient.final_confirmed_diagnosis.ilike("not malaria"))
        .scalar()
        or 0
    )
    pending_diagnosis = total_patients - malaria_confirmed - not_malaria
    total_medical_personnel = (
        db.query(func.count(User.id)).filter(User.user_type == UserType.MEDICAL_PERSONNEL).scalar() or 0
    )
    total_admins = (
        db.query(func.count(User.id)).filter(User.user_type == UserType.ADMIN).scalar() or 0
    )

    artifacts = get_predictor_artifacts()
    return DashboardStatsResponse(
        total_patients=total_patients,
        malaria_confirmed=malaria_confirmed,
        not_malaria=not_malaria,
        pending_diagnosis=max(pending_diagnosis, 0),
        total_medical_personnel=total_medical_personnel,
        total_admins=total_admins,
        model_loaded=artifacts is not None,
        model_name=artifacts.model_name if artifacts else None,
    )


@router.get("/patients", response_model=list[AdminPatientResponse])
def list_all_patients(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[AdminPatientResponse]:
    rows = db.query(Patient).order_by(Patient.created_at.desc()).all()
    return [_patient_to_admin_response(row) for row in rows]


@router.get("/patients/{patient_id}", response_model=AdminPatientResponse)
def get_patient_admin(
    patient_id: int,
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> AdminPatientResponse:
    patient = db.query(Patient).filter(Patient.id == patient_id).one_or_none()
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")
    return _patient_to_admin_response(patient)


@router.get("/users", response_model=list[AdminUserSummary])
def list_users(
    user_type: UserType | None = None,
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[AdminUserSummary]:
    query = db.query(User).order_by(User.created_at.desc())
    if user_type is not None:
        query = query.filter(User.user_type == user_type)
    return [AdminUserSummary.model_validate(row) for row in query.all()]
