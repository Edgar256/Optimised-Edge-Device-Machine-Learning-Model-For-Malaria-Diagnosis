"""Clinical feature engineering for triage-time malaria modelling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.common import is_missing
from src.utils.config import load_config
from src.utils.paths import resolve_path

ENGINEERED_FEATURE_COLUMNS = [
    "age_group",
    "pediatric_high_risk",
    "fever_severity",
    "symptom_count",
    "core_symptom_count",
    "transmission_risk_score",
    "is_rainy_season",
    "has_other_symptoms",
    "exposure_x_rainy_season",
    "household_history_x_child",
    "anemia_signs_x_pediatric",
    "fever_x_duration",
]

FEATURE_DEFINITIONS: list[dict[str, str]] = [
    {
        "name": "age_group",
        "type": "categorical",
        "definition": "child (<18), adult (18-59), senior (>=60)",
        "rationale": (
            "Paediatric and elderly patients differ in immune response, malaria presentation, "
            "and complication risk; age stratification is standard in Uganda health records."
        ),
    },
    {
        "name": "pediatric_high_risk",
        "type": "binary",
        "definition": "1 if age <= 4 years, else 0",
        "rationale": (
            "WHO prioritises under-5 malaria surveillance in endemic regions because young "
            "children face the highest morbidity and mortality burden."
        ),
    },
    {
        "name": "fever_severity",
        "type": "categorical",
        "definition": "none (no fever), acute (fever 0-2 days), prolonged (fever >=3 days)",
        "rationale": (
            "No body-temperature column exists; fever presence and duration proxy febrile "
            "severity. Prolonged fever raises clinical suspicion of malaria or complications."
        ),
    },
    {
        "name": "symptom_count",
        "type": "numeric",
        "definition": "Count of yes responses across fever, headache, chills, vomiting, fatigue",
        "rationale": (
            "Poly-symptom presentations are associated with higher malaria probability at triage."
        ),
    },
    {
        "name": "core_symptom_count",
        "type": "numeric",
        "definition": "Count of yes across fever, headache, vomiting only",
        "rationale": (
            "Excludes chills and fatigue which exceed 95% yes in this dataset and add little "
            "discriminative signal, likely due to form-default selection."
        ),
    },
    {
        "name": "transmission_risk_score",
        "type": "numeric",
        "definition": "Sum (0-4) of exposure risk, household malaria history, recent travel, rainy season",
        "rationale": (
            "Combines entomological exposure, household clustering, travel, and seasonal "
            "transmission into a single interpretable endemic-risk index."
        ),
    },
    {
        "name": "is_rainy_season",
        "type": "binary",
        "definition": "1 if season is Rainy, else 0",
        "rationale": (
            "Rainy season increases mosquito breeding and malaria transmission in Hoima District."
        ),
    },
    {
        "name": "has_other_symptoms",
        "type": "binary",
        "definition": "1 if free-text other symptoms field is non-empty",
        "rationale": (
            "Captures additional complaints (cough, diarrhoea, etc.) not covered by the "
            "structured yes/no symptom checklist."
        ),
    },
    {
        "name": "exposure_x_rainy_season",
        "type": "numeric",
        "definition": "exposure_risk_flag * is_rainy_season",
        "rationale": (
            "Mosquito-prone exposure during rainy season multiplies vector contact and "
            "transmission risk beyond either factor alone."
        ),
    },
    {
        "name": "household_history_x_child",
        "type": "numeric",
        "definition": "household_malaria_history_flag * is_child_flag",
        "rationale": (
            "Children in households with recent malaria face elevated reinfection risk due "
            "to shared exposure within the home."
        ),
    },
    {
        "name": "anemia_signs_x_pediatric",
        "type": "numeric",
        "definition": "anemia_signs_flag * pediatric_high_risk",
        "rationale": (
            "Anaemia signs in children under five are a severity marker in malaria-endemic "
            "areas; hemoglobin is not recorded so clinical anaemia signs are used as proxy."
        ),
    },
    {
        "name": "fever_x_duration",
        "type": "numeric",
        "definition": "fever_flag * fever_duration_days (NaN when duration missing)",
        "rationale": (
            "Combines fever presence with persistence; sustained febrile illness is more "
            "suggestive of malaria than isolated fever reporting without duration."
        ),
    },
]

EXCLUDED_FEATURE_DEFINITIONS: list[dict[str, str]] = [
    {
        "name": "Fever x RDT",
        "reason": "RDT is a diagnostic test result recorded during workup, not available at triage.",
    },
    {
        "name": "Microscopy x RDT",
        "reason": "Both microscopy and RDT are post-diagnosis columns that would leak the target.",
    },
    {
        "name": "Hemoglobin x Age",
        "reason": "No hemoglobin column exists; replaced by anemia_signs_x_pediatric interaction.",
    },
    {
        "name": "Temperature severity (Normal/Moderate/High)",
        "reason": "No body-temperature column exists; replaced by fever_severity proxy.",
    },
]


def _feature_engineering_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("feature_engineering", {})


def _yes_flag(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().eq("yes").astype(int)


def _assign_age_group(age: pd.Series, child_max: int, senior_min: int) -> pd.Series:
    groups = pd.Series(np.nan, index=age.index, dtype=object)
    groups[age < child_max + 1] = "child"
    groups[(age >= child_max + 1) & (age < senior_min)] = "adult"
    groups[age >= senior_min] = "senior"
    return groups


def _assign_fever_severity(
    fever: pd.Series,
    duration: pd.Series,
    acute_max_days: int,
    prolonged_min_days: int,
) -> pd.Series:
    fever_flag = fever.fillna("").astype(str).str.strip().str.lower().eq("yes")
    severity = pd.Series("none", index=fever.index, dtype=object)
    prolonged_mask = fever_flag & duration.notna() & (duration >= prolonged_min_days)
    severity = severity.mask(prolonged_mask, "prolonged")
    severity = severity.mask(fever_flag & ~prolonged_mask, "acute")
    return severity


def engineer_clinical_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Derive triage-safe clinical features from a cleaned processed dataset."""
    cfg = config or load_config()
    fe_cfg = _feature_engineering_config(cfg)

    _assert_no_leakage_columns(df, cfg)

    result = df.copy()
    age = pd.to_numeric(result["Age"], errors="coerce")
    fever_duration = pd.to_numeric(result["Fever Duration (Days)"], errors="coerce")

    child_max = int(fe_cfg.get("age_child_max", 17))
    senior_min = int(fe_cfg.get("age_senior_min", 60))
    pediatric_max = int(fe_cfg.get("pediatric_high_risk_max", 4))
    acute_max_days = int(fe_cfg.get("fever_acute_max_days", 2))
    prolonged_min_days = int(fe_cfg.get("fever_prolonged_min_days", 3))

    result["age_group"] = _assign_age_group(age, child_max, senior_min)
    result["pediatric_high_risk"] = (age <= pediatric_max).astype(int)
    result["fever_severity"] = _assign_fever_severity(
        result["Fever (Yes/No)"],
        fever_duration,
        acute_max_days,
        prolonged_min_days,
    )

    all_symptoms = fe_cfg.get("all_symptoms", [])
    core_symptoms = fe_cfg.get("core_symptoms", [])
    result["symptom_count"] = sum(_yes_flag(result[column]) for column in all_symptoms)
    result["core_symptom_count"] = sum(_yes_flag(result[column]) for column in core_symptoms)

    exposure_flag = _yes_flag(result["Exposure Risk (e.g., mosquito-prone area) (Yes/No)"])
    household_flag = _yes_flag(result["Household Malaria History (Yes/No)"])
    travel_flag = _yes_flag(result["Recent Travel (Yes/No)"])
    rainy_flag = result["Season of Visit (Dry/Rainy)"].fillna("").astype(str).str.strip().eq("Rainy").astype(int)
    anemia_flag = _yes_flag(result["Anemia Signs (Yes/No)"])
    fever_flag = _yes_flag(result["Fever (Yes/No)"])
    child_flag = (age < child_max + 1).astype(int)

    result["transmission_risk_score"] = exposure_flag + household_flag + travel_flag + rainy_flag
    result["is_rainy_season"] = rainy_flag
    result["has_other_symptoms"] = (~is_missing(result["Other Symptoms (Specify)"])).astype(int)
    result["exposure_x_rainy_season"] = exposure_flag * rainy_flag
    result["household_history_x_child"] = household_flag * child_flag
    result["anemia_signs_x_pediatric"] = anemia_flag * result["pediatric_high_risk"]
    result["fever_x_duration"] = fever_flag * fever_duration

    return result


def _assert_no_leakage_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    post_diagnosis = set(config["data"]["post_diagnosis_columns"])
    present = sorted(post_diagnosis & set(df.columns))
    if present:
        raise ValueError(
            "Processed dataset must not include post-diagnosis columns for feature engineering: "
            + ", ".join(present)
        )


def get_engineered_feature_names() -> list[str]:
    return list(ENGINEERED_FEATURE_COLUMNS)


@dataclass
class FeatureEngineeringResult:
    """Container for feature engineering outputs."""

    dataset: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for triage-time clinical feature engineering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ClinicalFeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer expects a pandas DataFrame.")
        cfg = self.config or load_config()
        _assert_no_leakage_columns(X, cfg)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer expects a pandas DataFrame.")
        return engineer_clinical_features(X, config=self.config or load_config())

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(get_engineered_feature_names(), dtype=object)


def load_processed_dataset(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the cleaned processed dataset."""
    cfg = config or load_config()
    fe_cfg = _feature_engineering_config(cfg)
    input_path = resolve_path(Path(cfg["paths"]["processed_data_dir"]) / fe_cfg.get("input_filename", "processed_dataset.csv"))
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at {input_path}. Run `python main.py preprocess` first."
        )
    return pd.read_csv(input_path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def _build_feature_distributions(df: pd.DataFrame) -> dict[str, Any]:
    distributions: dict[str, Any] = {}
    for column in ENGINEERED_FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column]
        if series.dtype == object or series.nunique(dropna=False) <= 10:
            distributions[column] = series.value_counts(dropna=False).to_dict()
        else:
            distributions[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(float(series.mean()), 2),
                "median": float(series.median()),
            }
    return distributions


def run_feature_engineering(
    config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> FeatureEngineeringResult:
    """Engineer features from the processed dataset and write the output CSV."""
    cfg = config or load_config()
    fe_cfg = _feature_engineering_config(cfg)
    input_df = load_processed_dataset(cfg)
    engineered_df = engineer_clinical_features(input_df, cfg)

    output_path = resolve_path(
        output_path
        or Path(cfg["paths"]["processed_data_dir"]) / fe_cfg.get("output_filename", "engineered_dataset.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output_path, index=False)

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(input_df),
        "output_rows": len(engineered_df),
        "engineered_feature_count": len(ENGINEERED_FEATURE_COLUMNS),
        "engineered_features": get_engineered_feature_names(),
        "output_path": _display_path(output_path),
        "distributions": _build_feature_distributions(engineered_df),
        "excluded_features": EXCLUDED_FEATURE_DEFINITIONS,
    }
    return FeatureEngineeringResult(dataset=engineered_df, stats=stats)


def write_feature_engineering_outputs(
    report_path: str | Path = "reports/feature_engineering_report.md",
    config: dict[str, Any] | None = None,
) -> FeatureEngineeringResult:
    """Run feature engineering and write the report."""
    from src.features.report import render_feature_engineering_report

    result = run_feature_engineering(config=config)
    report_file = resolve_path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_feature_engineering_report(result, config=config), encoding="utf-8")
    result.stats["report_path"] = _display_path(report_file)
    return result
