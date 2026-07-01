"""Tests for clinical feature engineering."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.engineering import (
    ClinicalFeatureEngineer,
    engineer_clinical_features,
    get_engineered_feature_names,
    run_feature_engineering,
    write_feature_engineering_outputs,
)
from src.utils.config import load_config


@pytest.fixture
def sample_processed_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Age": 17,
                "Fever (Yes/No)": "no",
                "Headache (Yes/No)": "yes",
                "Chills (Yes/No)": "yes",
                "Vomiting (Yes/No)": "no",
                "Fatigue (Yes/No)": "yes",
                "Fever Duration (Days)": 1.0,
                "Anemia Signs (Yes/No)": "no",
                "Season of Visit (Dry/Rainy)": "Rainy",
                "Recent Travel (Yes/No)": "no",
                "Exposure Risk (e.g., mosquito-prone area) (Yes/No)": "yes",
                "Household Malaria History (Yes/No)": "yes",
                "Other Symptoms (Specify)": "",
            }
        ]
    )


def test_age_group_boundaries() -> None:
    config = load_config()
    ages = pd.DataFrame({"Age": [17, 18, 59, 60]})
    for column in [
        "Fever (Yes/No)",
        "Headache (Yes/No)",
        "Chills (Yes/No)",
        "Vomiting (Yes/No)",
        "Fatigue (Yes/No)",
        "Anemia Signs (Yes/No)",
        "Season of Visit (Dry/Rainy)",
        "Recent Travel (Yes/No)",
        "Exposure Risk (e.g., mosquito-prone area) (Yes/No)",
        "Household Malaria History (Yes/No)",
        "Other Symptoms (Specify)",
        "Fever Duration (Days)",
    ]:
        ages[column] = "no" if "Fever" not in column else ""
    ages["Season of Visit (Dry/Rainy)"] = "Dry"

    engineered = engineer_clinical_features(ages, config)
    assert engineered.loc[0, "age_group"] == "child"
    assert engineered.loc[1, "age_group"] == "adult"
    assert engineered.loc[2, "age_group"] == "adult"
    assert engineered.loc[3, "age_group"] == "senior"


def test_symptom_count_matches_manual(sample_processed_row: pd.DataFrame) -> None:
    engineered = engineer_clinical_features(sample_processed_row, load_config())
    assert engineered.loc[0, "symptom_count"] == 3
    assert engineered.loc[0, "core_symptom_count"] == 1


def test_fever_severity_none_when_no_fever(sample_processed_row: pd.DataFrame) -> None:
    engineered = engineer_clinical_features(sample_processed_row, load_config())
    assert engineered.loc[0, "fever_severity"] == "none"


def test_fever_severity_prolonged() -> None:
    config = load_config()
    row = pd.DataFrame(
        [
            {
                "Age": 25,
                "Fever (Yes/No)": "yes",
                "Headache (Yes/No)": "no",
                "Chills (Yes/No)": "no",
                "Vomiting (Yes/No)": "no",
                "Fatigue (Yes/No)": "no",
                "Fever Duration (Days)": 3.0,
                "Anemia Signs (Yes/No)": "no",
                "Season of Visit (Dry/Rainy)": "Dry",
                "Recent Travel (Yes/No)": "no",
                "Exposure Risk (e.g., mosquito-prone area) (Yes/No)": "no",
                "Household Malaria History (Yes/No)": "no",
                "Other Symptoms (Specify)": "",
            }
        ]
    )
    engineered = engineer_clinical_features(row, config)
    assert engineered.loc[0, "fever_severity"] == "prolonged"


def test_no_post_diagnosis_columns_in_engineered_features() -> None:
    config = load_config()
    post_diagnosis = set(config["data"]["post_diagnosis_columns"])
    engineered_names = set(get_engineered_feature_names())
    assert engineered_names.isdisjoint(post_diagnosis)


def test_leakage_columns_raise_error() -> None:
    config = load_config()
    leaky = pd.DataFrame(
        {
            "Age": [10],
            "Rapid Diagnostic Test (RDT) Result (Positive/Negative)": ["positive"],
            "Fever (Yes/No)": ["yes"],
            "Headache (Yes/No)": ["no"],
            "Chills (Yes/No)": ["no"],
            "Vomiting (Yes/No)": ["no"],
            "Fatigue (Yes/No)": ["no"],
            "Fever Duration (Days)": [1],
            "Anemia Signs (Yes/No)": ["no"],
            "Season of Visit (Dry/Rainy)": ["Dry"],
            "Recent Travel (Yes/No)": ["no"],
            "Exposure Risk (e.g., mosquito-prone area) (Yes/No)": ["no"],
            "Household Malaria History (Yes/No)": ["no"],
            "Other Symptoms (Specify)": [""],
        }
    )
    with pytest.raises(ValueError, match="post-diagnosis"):
        engineer_clinical_features(leaky, config)


def test_clinical_feature_engineer_sklearn_interface(sample_processed_row: pd.DataFrame) -> None:
    transformer = ClinicalFeatureEngineer()
    output = transformer.fit_transform(sample_processed_row)
    assert "transmission_risk_score" in output.columns


def test_run_feature_engineering_writes_output(tmp_path: Path) -> None:
    config = load_config()
    resolve_processed_for_test(config, tmp_path)
    output_path = tmp_path / "engineered_dataset.csv"

    result = run_feature_engineering(
        config=config,
        output_path=output_path,
    )
    assert output_path.is_file()
    assert result.stats["output_rows"] > 0
    assert all(feature in result.dataset.columns for feature in get_engineered_feature_names())


def test_write_feature_engineering_outputs_creates_report(tmp_path: Path) -> None:
    config = load_config()
    resolve_processed_for_test(config, tmp_path)
    report_path = tmp_path / "feature_engineering_report.md"

    result = write_feature_engineering_outputs(report_path=report_path, config=config)
    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "Feature Engineering Report" in content
    assert "Fever x RDT" in content
    assert result.stats["engineered_feature_count"] == 12


def resolve_processed_for_test(config: dict, tmp_path: Path) -> Path:
    """Point config at real processed data or copy it into tmp_path."""
    from src.utils.paths import resolve_path

    source = resolve_path(Path(config["paths"]["processed_data_dir"]) / config["feature_engineering"]["input_filename"])
    if not source.is_file():
        pytest.skip("processed_dataset.csv not found; run preprocess first")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    dest = processed_dir / "processed_dataset.csv"
    dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    config["paths"]["processed_data_dir"] = str(processed_dir)
    return dest
