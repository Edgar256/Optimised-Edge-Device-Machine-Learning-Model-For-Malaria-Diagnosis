"""Tests for training-artifact cleanup."""

from pathlib import Path

from src.utils.cleanup import clean_training_artifacts


def _artifact_config(tmp_path: Path) -> dict:
    paths = {
        "models_dir": str(tmp_path / "models"),
        "results_dir": str(tmp_path / "results"),
        "figures_dir": str(tmp_path / "figures"),
        "reports_dir": str(tmp_path / "reports"),
        "processed_data_dir": str(tmp_path / "data" / "processed"),
        "raw_data_dir": str(tmp_path / "data" / "raw"),
    }
    for key, value in paths.items():
        Path(value).mkdir(parents=True, exist_ok=True)
        (Path(value) / ".gitkeep").write_text("", encoding="utf-8")
    return {"paths": paths}


def test_clean_training_artifacts_removes_outputs_keeps_gitkeep(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    models = Path(config["paths"]["models_dir"])
    results = Path(config["paths"]["results_dir"])
    reports = Path(config["paths"]["reports_dir"])
    processed = Path(config["paths"]["processed_data_dir"])

    (models / "baseline").mkdir()
    (models / "baseline" / "gradient_boosting.joblib").write_text("model", encoding="utf-8")
    (results / "baseline_ranking.csv").write_text("rank\n", encoding="utf-8")
    (reports / "baseline_training_report.md").write_text("# report\n", encoding="utf-8")
    (processed / "processed_dataset.csv").write_text("x\n", encoding="utf-8")

    result = clean_training_artifacts(config=config)

    assert not (models / "baseline" / "gradient_boosting.joblib").exists()
    assert not (results / "baseline_ranking.csv").exists()
    assert not (reports / "baseline_training_report.md").exists()
    assert (models / ".gitkeep").exists()
    assert (results / ".gitkeep").exists()
    assert (processed / "processed_dataset.csv").exists()
    assert len(result.deleted) >= 3


def test_clean_training_artifacts_include_processed(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    processed = Path(config["paths"]["processed_data_dir"])
    (processed / "processed_dataset.csv").write_text("x\n", encoding="utf-8")

    clean_training_artifacts(config=config, include_processed=True)

    assert not (processed / "processed_dataset.csv").exists()
    assert (processed / ".gitkeep").exists()


def test_clean_training_artifacts_dry_run_does_not_delete(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    results = Path(config["paths"]["results_dir"])
    target = results / "baseline_ranking.csv"
    target.write_text("rank\n", encoding="utf-8")

    result = clean_training_artifacts(config=config, dry_run=True)

    assert target.exists()
    assert result.dry_run is True
    assert any("baseline_ranking.csv" in path for path in result.deleted)
