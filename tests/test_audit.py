"""Tests for the read-only data quality audit."""

from pathlib import Path

from src.preprocessing.audit import run_audit, write_audit_outputs


def test_run_audit_returns_expected_shape() -> None:
    audit = run_audit()
    assert audit.findings["shape"]["rows"] == 291
    assert audit.findings["shape"]["columns"] == 37
    assert len(audit.summary) == 37


def test_write_audit_outputs_creates_files(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    summary_path = tmp_path / "summary.csv"
    write_audit_outputs(report_path=report_path, summary_path=summary_path)

    assert report_path.is_file()
    assert summary_path.is_file()
    assert "Data Quality Report" in report_path.read_text(encoding="utf-8")
