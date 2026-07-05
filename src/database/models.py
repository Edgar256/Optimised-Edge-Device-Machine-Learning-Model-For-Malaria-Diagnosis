"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.enums import UserType
from src.database.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    first_name: Mapped[str] = mapped_column(String(120), nullable=False)
    last_name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    phone: Mapped[str] = mapped_column(String(40), nullable=False)
    job_title: Mapped[str] = mapped_column(String(120), nullable=False)
    health_facility_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_type: Mapped[UserType] = mapped_column(
        Enum(UserType, name="user_type", native_enum=False, length=32),
        nullable=False,
        default=UserType.MEDICAL_PERSONNEL,
        server_default=UserType.MEDICAL_PERSONNEL.value,
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    patients: Mapped[list[Patient]] = relationship(back_populates="created_by_user")


class Patient(Base):
    """Patient visit record aligned with the Kobo extraction form."""

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # Kobo / form timestamps
    start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    patient_id_anonymous_code: Mapped[str | None] = mapped_column(String(120), nullable=True)
    health_facility_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    date_of_visit: Mapped[date | None] = mapped_column(Date, nullable=True)
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    geographical_zone: Mapped[str | None] = mapped_column(String(255), nullable=True)

    fever: Mapped[str | None] = mapped_column(String(10), nullable=True)
    headache: Mapped[str | None] = mapped_column(String(10), nullable=True)
    chills: Mapped[str | None] = mapped_column(String(10), nullable=True)
    vomiting: Mapped[str | None] = mapped_column(String(10), nullable=True)
    fatigue: Mapped[str | None] = mapped_column(String(10), nullable=True)
    other_symptoms_specify: Mapped[str | None] = mapped_column(Text, nullable=True)
    fever_duration_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    anemia_signs: Mapped[str | None] = mapped_column(String(10), nullable=True)
    season_of_visit: Mapped[str | None] = mapped_column(String(20), nullable=True)
    recent_travel: Mapped[str | None] = mapped_column(String(10), nullable=True)
    exposure_risk: Mapped[str | None] = mapped_column(String(10), nullable=True)
    household_malaria_history: Mapped[str | None] = mapped_column(String(10), nullable=True)

    rdt_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    microscopy_result: Mapped[str | None] = mapped_column(String(20), nullable=True)
    parasite_density: Mapped[str | None] = mapped_column(String(120), nullable=True)
    final_confirmed_diagnosis: Mapped[str | None] = mapped_column(String(80), nullable=True)
    treatment_given: Mapped[str | None] = mapped_column(Text, nullable=True)
    additional_clinical_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Kobo metadata
    kobo_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    kobo_uuid: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    submission_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    validation_status: Mapped[str | None] = mapped_column(String(120), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(String(80), nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    form_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    tags: Mapped[str | None] = mapped_column(String(255), nullable=True)
    meta_root_uuid: Mapped[str | None] = mapped_column(String(80), nullable=True)
    kobo_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Mobile capture
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    created_by_user: Mapped[User] = relationship(back_populates="patients")
