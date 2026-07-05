"""Patient records package."""

from src.patients.routes import router as patients_router
from src.patients.schemas import PatientCreateRequest, PatientResponse, PatientUpdateRequest

__all__ = [
    "PatientCreateRequest",
    "PatientResponse",
    "PatientUpdateRequest",
    "patients_router",
]
