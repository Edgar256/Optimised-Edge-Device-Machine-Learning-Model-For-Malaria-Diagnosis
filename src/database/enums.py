"""Database enums."""

from __future__ import annotations

import enum


class UserType(str, enum.Enum):
    ADMIN = "ADMIN"
    MEDICAL_PERSONNEL = "MEDICAL_PERSONNEL"
