"""Tests for the preprocessing pipeline."""

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.common import load_raw_dataframe
from src.preprocessing.pipeline import build_preprocessing_pipeline, run_preprocessing, write_preprocessing_outputs
from src.preprocessing.transformers import canonicalize_facility, canonicalize_zone
from src.utils.config import load_config


def test_canonicalize_facility_maps_variants() -> None:
    assert canonicalize_facility("Mbaraara21") == "mbaraara"
    assert canonicalize_facility("kibaire") == "kibaire"


def test_canonicalize_zone_maps_typos() -> None:
    assert canonicalize_zone("kib aire") == "kibaire"
    assert canonicalize_zone("Mbaraarac") == "mbaraara"


def test_build_preprocessing_pipeline_has_column_transformer() -> None:
    pipeline = build_preprocessing_pipeline(load_config())
    assert "column_transformer" in pipeline.named_steps
    assert "row_filter" in pipeline.named_steps


def test_run_preprocessing_reduces_rows_and_writes_artifacts(tmp_path: Path) -> None:
    config = load_config()
    processed_path = tmp_path / "processed_dataset.csv"
    pipeline_path = tmp_path / "preprocessing_pipeline.joblib"

    result = run_preprocessing(
        config=config,
        processed_dataset_path=processed_path,
        pipeline_path=pipeline_path,
    )

    assert processed_path.is_file()
    assert pipeline_path.is_file()
    assert result.stats["initial_rows"] == 291
    assert result.stats["final_rows"] < result.stats["initial_rows"]
    assert "target_binary" in result.cleaned_dataset.columns
    assert len(result.feature_names) > 0

    loaded = joblib.load(pipeline_path)
    raw_df, _ = load_raw_dataframe(config)
    transformed = loaded.transform(raw_df)
    assert transformed.shape[0] == len(result.cleaned_dataset)


def test_write_preprocessing_outputs_creates_report(tmp_path: Path, monkeypatch) -> None:
    config = load_config()
    processed_dir = tmp_path / "processed"
    report_path = tmp_path / "preprocessing_report.md"
    processed_dir.mkdir()

    config["paths"]["processed_data_dir"] = str(processed_dir)

    result = write_preprocessing_outputs(report_path=report_path, config=config)
    assert report_path.is_file()
    assert "Preprocessing Report" in report_path.read_text(encoding="utf-8")
    assert result.stats["final_rows"] > 0
