"""Patient record API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_medical_personnel
from src.database.models import Patient, User
from src.database.session import get_db
from src.patients.schemas import PatientCreateRequest, PatientResponse, PatientUpdateRequest

router = APIRouter(prefix="/v1/patients", tags=["patients"])


def _apply_patient_fields(
    patient: Patient, payload: PatientCreateRequest | PatientUpdateRequest
) -> None:
    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(patient, field, value)


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    payload: PatientCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_medical_personnel),
) -> PatientResponse:
    patient = Patient(created_by_user_id=current_user.id)
    _apply_patient_fields(patient, payload)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return PatientResponse.model_validate(patient)


@router.get("", response_model=list[PatientResponse])
def list_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_medical_personnel),
) -> list[PatientResponse]:
    rows = (
        db.query(Patient)
        .filter(Patient.created_by_user_id == current_user.id)
        .order_by(Patient.created_at.desc())
        .all()
    )
    return [PatientResponse.model_validate(row) for row in rows]


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_medical_personnel),
) -> PatientResponse:
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.created_by_user_id == current_user.id)
        .one_or_none()
    )
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")
    return PatientResponse.model_validate(patient)


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int,
    payload: PatientUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_medical_personnel),
) -> PatientResponse:
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.created_by_user_id == current_user.id)
        .one_or_none()
    )
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")
    _apply_patient_fields(patient, payload)
    db.commit()
    db.refresh(patient)
    return PatientResponse.model_validate(patient)


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_medical_personnel),
) -> None:
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.created_by_user_id == current_user.id)
        .one_or_none()
    )
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")
    db.delete(patient)
    db.commit()
