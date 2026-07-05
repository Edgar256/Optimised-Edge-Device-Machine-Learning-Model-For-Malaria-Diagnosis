"""Authentication Pydantic schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from src.database.enums import UserType


class UserSignupRequest(BaseModel):
    """Mobile app registration for medical personnel."""

    model_config = ConfigDict(extra="forbid")

    first_name: str = Field(..., min_length=1, max_length=120)
    last_name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    phone: str = Field(..., min_length=5, max_length=40)
    job_title: str = Field(..., min_length=1, max_length=120)
    health_facility_name: str | None = Field(default=None, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class AdminSignupRequest(UserSignupRequest):
    """Admin console registration (requires server secret)."""

    admin_registration_secret: str = Field(..., min_length=8, max_length=256)


class UserLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)
    remember_me: bool = False


class UserResponse(BaseModel):
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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    remember_me: bool
    user: UserResponse
