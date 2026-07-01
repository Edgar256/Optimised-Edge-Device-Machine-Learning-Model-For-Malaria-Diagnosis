"""Smoke tests for project configuration and raw data layout."""

from pathlib import Path

import pandas as pd
import pytest

from src.utils.config import get_raw_data_path, load_config
from src.utils.paths import find_project_root


@pytest.fixture
def config() -> dict:
    return load_config()


def test_project_root_contains_expected_directories() -> None:
    root = find_project_root(Path(__file__))
    for name in ("config", "data/raw", "src", "tests"):
        assert (root / name).exists(), f"Missing expected path: {name}"


def test_raw_dataset_is_readable(config: dict) -> None:
    raw_path = get_raw_data_path(config)
    assert raw_path.is_file()

    df = pd.read_csv(
        raw_path,
        sep=config["data"]["csv_separator"],
        encoding=config["data"]["encoding"],
        nrows=5,
    )
    assert config["data"]["target_column"] in df.columns


def test_config_lists_disjoint_column_groups(config: dict) -> None:
    data_cfg = config["data"]
    groups = [
        set(data_cfg["identifier_columns"]),
        set(data_cfg["metadata_columns"]),
        set(data_cfg["post_diagnosis_columns"]),
        set(data_cfg["clinical_feature_columns"]),
        {data_cfg["target_column"]},
    ]
    all_columns = [column for group in groups for column in group]
    assert len(all_columns) == len(set(all_columns))
