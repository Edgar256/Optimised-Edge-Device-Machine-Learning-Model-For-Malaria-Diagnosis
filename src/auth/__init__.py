"""Authentication package."""

from src.auth.dependencies import get_current_admin, get_current_medical_personnel, get_current_user
from src.database.enums import UserType
from src.auth.routes import router as auth_router
from src.auth.schemas import (
    AdminSignupRequest,
    TokenResponse,
    UserLoginRequest,
    UserResponse,
    UserSignupRequest,
)

__all__ = [
    "AdminSignupRequest",
    "TokenResponse",
    "UserLoginRequest",
    "UserResponse",
    "UserSignupRequest",
    "UserType",
    "auth_router",
    "get_current_admin",
    "get_current_medical_personnel",
    "get_current_user",
]
