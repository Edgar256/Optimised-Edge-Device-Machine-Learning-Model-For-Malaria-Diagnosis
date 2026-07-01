"""Shared preprocessing helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import get_raw_data_path, load_config


def is_missing(series: pd.Series) -> pd.Series:
    """Treat NaN, empty strings, and whitespace-only values as missing."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.isna() | series.astype(str).str.strip().eq("")
    return series.isna()


def load_raw_dataframe(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw CSV using project configuration."""
    cfg = config or load_config()
    data_cfg = cfg["data"]
    path = get_raw_data_path(cfg)
    df = pd.read_csv(path, sep=data_cfg["csv_separator"], encoding=data_cfg["encoding"])
    return df, cfg


def yes_no_columns() -> list[str]:
    return [
        "Fever (Yes/No)",
        "Headache (Yes/No)",
        "Chills (Yes/No)",
        "Vomiting (Yes/No)",
        "Fatigue (Yes/No)",
        "Anemia Signs (Yes/No)",
        "Recent Travel (Yes/No)",
        "Exposure Risk (e.g., mosquito-prone area) (Yes/No)",
        "Household Malaria History (Yes/No)",
    ]


def numeric_feature_columns() -> list[str]:
    return ["Age", "Fever Duration (Days)"]


def categorical_feature_columns() -> list[str]:
    return [
        "Health Facility Name",
        "Gender (Male/Female)",
        "Geographical Zone",
        *yes_no_columns(),
        "Season of Visit (Dry/Rainy)",
        "Other Symptoms (Specify)",
        "visit_month",
    ]


def model_feature_columns() -> list[str]:
    return [*numeric_feature_columns(), *categorical_feature_columns()]
