"""Custom sklearn transformers for malaria clinical data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.common import is_missing, yes_no_columns


def _is_blank(value: Any) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _normalize_facility_cell(value: Any) -> Any:
    if _is_blank(value):
        return np.nan
    return canonicalize_facility(str(value))


def _normalize_zone_cell(value: Any) -> Any:
    if _is_blank(value):
        return np.nan
    return canonicalize_zone(str(value))


def canonicalize_facility(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "nan"}:
        return np.nan  # type: ignore[return-value]
    if "test" in normalized:
        return "test_facility"
    if "kasomoro" in normalized:
        return "kasomoro_health_centre"
    if "mbaraara" in normalized or "mbaraar" in normalized or "mnaraara" in normalized:
        return "mbaraara"
    if "kibaire" in normalized or "kib aire" in normalized:
        return "kibaire"
    return normalized


def canonicalize_zone(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "nan"}:
        return np.nan  # type: ignore[return-value]
    if "mbaraara" in normalized or "mbaraar" in normalized or "mnaraara" in normalized:
        return "mbaraara"
    if "kibaire" in normalized or "kib aire" in normalized:
        return "kibaire"
    return normalized


class RowFilterTransformer(BaseEstimator, TransformerMixin):
    """Drop duplicate, invalid, and unlabelled rows while preserving the index."""

    def __init__(
        self,
        target_column: str,
        clinical_columns: list[str],
        test_patient_ids: list[str] | None = None,
        id_column: str = "Patient ID (Anonymous Code)",
    ) -> None:
        self.target_column = target_column
        self.clinical_columns = clinical_columns
        self.test_patient_ids = [value.lower() for value in (test_patient_ids or [])]
        self.id_column = id_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RowFilterTransformer:
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        filtered = X.copy()
        initial_rows = len(filtered)

        filtered = filtered.drop_duplicates()
        duplicate_rows_removed = initial_rows - len(filtered)

        if "_uuid" in filtered.columns:
            before_uuid = len(filtered)
            filtered = filtered.drop_duplicates(subset="_uuid", keep="first")
            duplicate_uuid_removed = before_uuid - len(filtered)
        else:
            duplicate_uuid_removed = 0

        target = filtered[self.target_column]
        missing_target_mask = is_missing(target)
        missing_target_removed = int(missing_target_mask.sum())
        filtered = filtered.loc[~missing_target_mask]

        if self.id_column in filtered.columns:
            patient_ids = filtered[self.id_column].fillna("").astype(str).str.strip().str.lower()
            test_mask = patient_ids.isin(self.test_patient_ids)
            test_rows_removed = int(test_mask.sum())
            filtered = filtered.loc[~test_mask]
        else:
            test_rows_removed = 0

        clinical_cols = [column for column in self.clinical_columns if column in filtered.columns]
        empty_clinical_mask = filtered[clinical_cols].apply(is_missing).all(axis=1)
        empty_clinical_removed = int(empty_clinical_mask.sum())
        filtered = filtered.loc[~empty_clinical_mask]

        self.filter_stats_ = {
            "initial_rows": initial_rows,
            "duplicate_rows_removed": duplicate_rows_removed,
            "duplicate_uuid_removed": duplicate_uuid_removed,
            "missing_target_removed": missing_target_removed,
            "test_rows_removed": test_rows_removed,
            "empty_clinical_removed": empty_clinical_removed,
            "final_rows": len(filtered),
        }
        self.retained_index_ = filtered.index
        return filtered.reset_index(drop=True)

    def _validate_input(self, X: pd.DataFrame) -> None:
        if self.target_column not in X.columns:
            raise ValueError(f"Target column not found: {self.target_column}")


class CategoryNormalizerTransformer(BaseEstimator, TransformerMixin):
    """Normalize inconsistent categorical text values."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CategoryNormalizerTransformer:
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        normalized = X.copy()

        for column in yes_no_columns():
            if column in normalized.columns:
                values = normalized[column].astype(str).str.strip().str.lower()
                values = values.replace({"": np.nan, "nan": np.nan})
                normalized[column] = values

        if "Gender (Male/Female)" in normalized.columns:
            gender = normalized["Gender (Male/Female)"].astype(str).str.strip().str.lower()
            gender = gender.replace({"": np.nan, "nan": np.nan})
            normalized["Gender (Male/Female)"] = gender

        if "Season of Visit (Dry/Rainy)" in normalized.columns:
            season = normalized["Season of Visit (Dry/Rainy)"].astype(str).str.strip()
            season = season.replace({"": np.nan, "nan": np.nan})
            normalized["Season of Visit (Dry/Rainy)"] = season

        if "Health Facility Name" in normalized.columns:
            normalized["Health Facility Name"] = normalized["Health Facility Name"].map(_normalize_facility_cell)

        if "Geographical Zone" in normalized.columns:
            normalized["Geographical Zone"] = normalized["Geographical Zone"].map(_normalize_zone_cell)

        if "Other Symptoms (Specify)" in normalized.columns:
            symptoms = normalized["Other Symptoms (Specify)"].astype(str).str.strip().str.lower()
            symptoms = symptoms.replace({"": np.nan, "nan": np.nan})
            normalized["Other Symptoms (Specify)"] = symptoms

        return normalized

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CategoryNormalizerTransformer expects a pandas DataFrame.")


class ImpossibleValueTransformer(BaseEstimator, TransformerMixin):
    """Replace clinically impossible numeric values with missing values."""

    def __init__(
        self,
        age_column: str = "Age",
        fever_duration_column: str = "Fever Duration (Days)",
        min_age: float = 0.0,
        max_age: float = 120.0,
        min_fever_duration: float = 0.0,
        max_fever_duration: float = 60.0,
    ) -> None:
        self.age_column = age_column
        self.fever_duration_column = fever_duration_column
        self.min_age = min_age
        self.max_age = max_age
        self.min_fever_duration = min_fever_duration
        self.max_fever_duration = max_fever_duration

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ImpossibleValueTransformer:
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        corrected = X.copy()
        corrections = {"age_out_of_range": 0, "fever_duration_out_of_range": 0}

        if self.age_column in corrected.columns:
            age = pd.to_numeric(corrected[self.age_column], errors="coerce")
            invalid_age = age.notna() & ((age < self.min_age) | (age > self.max_age))
            corrections["age_out_of_range"] = int(invalid_age.sum())
            age = age.mask(invalid_age)
            corrected[self.age_column] = age

        if self.fever_duration_column in corrected.columns:
            fever_duration = pd.to_numeric(corrected[self.fever_duration_column], errors="coerce")
            invalid_duration = fever_duration.notna() & (
                (fever_duration < self.min_fever_duration) | (fever_duration > self.max_fever_duration)
            )
            corrections["fever_duration_out_of_range"] = int(invalid_duration.sum())
            fever_duration = fever_duration.mask(invalid_duration)
            corrected[self.fever_duration_column] = fever_duration

        self.correction_stats_ = corrections
        return corrected

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ImpossibleValueTransformer expects a pandas DataFrame.")


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract model-friendly date features from visit dates."""

    def __init__(self, date_column: str = "Date of Visit", output_column: str = "visit_month") -> None:
        self.date_column = date_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DateFeatureExtractor:
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        enriched = X.copy()
        if self.date_column not in enriched.columns:
            return enriched

        visit_dates = pd.to_datetime(enriched[self.date_column], errors="coerce")
        enriched[self.output_column] = visit_dates.dt.month_name().str.lower()
        enriched[self.output_column] = enriched[self.output_column].replace({"nan": np.nan})
        return enriched

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DateFeatureExtractor expects a pandas DataFrame.")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select the model feature columns used by the ColumnTransformer."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureSelector:
        missing = sorted(set(self.feature_columns) - set(X.columns))
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.feature_columns].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(self.feature_columns, dtype=object)
