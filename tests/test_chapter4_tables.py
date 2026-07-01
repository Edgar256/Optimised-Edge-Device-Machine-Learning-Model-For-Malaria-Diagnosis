"""Tests for Chapter 4 thesis table generation."""

import pytest

from src.evaluation.chapter4_tables import (
    TABLE_SPECS,
    build_table_4_1_dataset_summary,
    build_table_4_2_missing_values,
    build_table_4_4_baseline_results,
    generate_chapter4_tables,
    write_chapter4_tables,
)


def test_table_specs_cover_all_seven_tables() -> None:
    expected = {f"table_4_{index}_{name}" for index, name in [
        (1, "dataset_summary"),
        (2, "missing_values"),
        (3, "feature_statistics"),
        (4, "baseline_results"),
        (5, "optimized_results"),
        (6, "model_comparison"),
        (7, "edge_performance"),
    ]}
    assert set(TABLE_SPECS) == expected
    for spec in TABLE_SPECS.values():
        assert spec["title"].startswith("Table 4.")
        assert len(spec["caption"]) > 20


def test_build_table_4_1_has_core_metrics() -> None:
    table = build_table_4_1_dataset_summary()
    metrics = set(table["metric"])
    assert "Raw records" in metrics
    assert "Malaria cases" in metrics
    assert "Encoded model features" in metrics


def test_build_table_4_2_missing_values_sorted() -> None:
    table = build_table_4_2_missing_values()
    assert "variable" in table.columns
    assert table["missing_percent"].is_monotonic_decreasing


def test_build_table_4_4_baseline_results() -> None:
    try:
        table = build_table_4_4_baseline_results()
    except FileNotFoundError:
        pytest.skip("baseline results not found")
    assert "model" in table.columns
    assert len(table) >= 1


def test_generate_all_chapter4_tables() -> None:
    try:
        result = generate_chapter4_tables()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert len(result.tables) == 7
    assert len(result.captions) == 7


def test_write_chapter4_tables_exports_csv_and_png(tmp_path) -> None:
    try:
        result = write_chapter4_tables(output_dir=tmp_path / "chapter4")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    for key in TABLE_SPECS:
        assert (tmp_path / "chapter4" / "tables" / f"{key}.csv").is_file()
        assert (tmp_path / "chapter4" / "figures" / f"{key}.png").is_file()
    assert (tmp_path / "chapter4" / "manifest.json").is_file()
    assert result.stats["table_count"] == 7
