"""Authentication API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.auth.dependencies import get_current_admin, get_current_user
from src.database.enums import UserType
from src.auth.schemas import (
    AdminSignupRequest,
    TokenResponse,
    UserLoginRequest,
    UserResponse,
    UserSignupRequest,
)
from src.auth.security import create_access_token, hash_password, verify_password
from src.database.models import User
from src.database.session import get_db
from src.utils.env import get_admin_registration_secret

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _build_user(payload: UserSignupRequest, *, user_type: UserType) -> User:
    return User(
        first_name=payload.first_name.strip(),
        last_name=payload.last_name.strip(),
        email=payload.email.lower().strip(),
        phone=payload.phone.strip(),
        job_title=payload.job_title.strip(),
        health_facility_name=(
            payload.health_facility_name.strip() if payload.health_facility_name else None
        ),
        user_type=user_type,
        password_hash=hash_password(payload.password),
    )


def _issue_token(user: User, *, remember_me: bool) -> TokenResponse:
    token, expires_in = create_access_token(subject=user.email, remember_me=remember_me)
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        remember_me=remember_me,
        user=UserResponse.model_validate(user),
    )


def _authenticate_user(
    db: Session,
    payload: UserLoginRequest,
    *,
    expected_type: UserType,
    wrong_type_detail: str,
) -> User:
    user = db.query(User).filter(User.email == payload.email.lower().strip()).one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if user.user_type != expected_type:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=wrong_type_detail,
        )
    return user


def _persist_user(db: Session, user: User) -> User:
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        ) from exc
    db.refresh(user)
    return user


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup_medical_personnel(
    payload: UserSignupRequest, db: Session = Depends(get_db)
) -> TokenResponse:
    """Register medical personnel for the mobile app."""
    user = _persist_user(db, _build_user(payload, user_type=UserType.MEDICAL_PERSONNEL))
    return _issue_token(user, remember_me=True)


@router.post("/login", response_model=TokenResponse)
def login_medical_personnel(payload: UserLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Login for medical personnel (mobile app)."""
    user = _authenticate_user(
        db,
        payload,
        expected_type=UserType.MEDICAL_PERSONNEL,
        wrong_type_detail="Use the admin login endpoint for administrator accounts.",
    )
    return _issue_token(user, remember_me=payload.remember_me)


@router.post("/admin/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register_admin(payload: AdminSignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Register an administrator account (admin console)."""
    configured_secret = get_admin_registration_secret()
    if not configured_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin registration is not configured. Set ADMIN_REGISTRATION_SECRET in .env.",
        )
    if payload.admin_registration_secret != configured_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin registration secret.",
        )

    user = _persist_user(db, _build_user(payload, user_type=UserType.ADMIN))
    return _issue_token(user, remember_me=True)


@router.post("/admin/login", response_model=TokenResponse)
def login_admin(payload: UserLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Login for administrators (admin console)."""
    user = _authenticate_user(
        db,
        payload,
        expected_type=UserType.ADMIN,
        wrong_type_detail="Use the mobile app login endpoint for medical personnel accounts.",
    )
    return _issue_token(user, remember_me=payload.remember_me)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.get("/admin/me", response_model=UserResponse)
def admin_me(current_admin: User = Depends(get_current_admin)) -> UserResponse:
    return UserResponse.model_validate(current_admin)
