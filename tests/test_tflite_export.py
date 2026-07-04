"""Tests for versioned TFLite export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.deployment.tflite_export import (
    PARITY_TOLERANCE,
    bump_semver,
    convert_lr_pipeline_to_tflite,
    export_tflite_model,
    predict_tflite_proba,
)


def test_bump_semver() -> None:
    assert bump_semver("1.2.3", "patch") == "1.2.4"
    assert bump_semver("1.2.3", "minor") == "1.3.0"
    assert bump_semver("1.2.3", "major") == "2.0.0"
    assert bump_semver("v0.0.0", "patch") == "0.0.1"


def test_convert_lr_pipeline_parity() -> None:
    pytest.importorskip("tensorflow")

    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    tflite_bytes = convert_lr_pipeline_to_tflite(scaler, model)
    sample = X[:16]
    sklearn_proba = pipeline.predict_proba(sample)[:, 1]
    tflite_proba = predict_tflite_proba(tflite_bytes, sample)
    assert float(np.max(np.abs(sklearn_proba - tflite_proba))) <= PARITY_TOLERANCE


def test_export_tflite_model_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tensorflow")

    from src.deployment import tflite_export as module
    from src.utils.config import load_config

    cfg = load_config()
    cfg = {
        **cfg,
        "tflite": {"output_dir": str(tmp_path / "models" / "tflite"), "min_app_version": "1.0.0"},
    }

    try:
        result = export_tflite_model(
            version="1.0.0",
            model_name="logistic_regression",
            config=cfg,
            release_notes="Initial mobile release",
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    except ValueError as exc:
        if "supports only" in str(exc):
            pytest.skip(str(exc))
        raise

    assert result.model_path.is_file()
    assert result.feature_spec_path.is_file()
    assert result.metadata_path.is_file()
    assert result.manifest_path.is_file()
    assert result.max_parity_error <= PARITY_TOLERANCE

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["latest_version"] == "1.0.0"
    assert manifest["model_name"] == "logistic_regression"
    assert "1.0.0" in manifest["releases"]
    assert manifest["releases"]["1.0.0"]["model_sha256"] == result.stats["model_sha256"]

    feature_spec = json.loads(result.feature_spec_path.read_text(encoding="utf-8"))
    assert feature_spec["n_features"] == result.stats["n_features"]
    assert len(feature_spec["feature_names"]) == feature_spec["n_features"]

    # Second export with patch bump updates latest_version.
    result2 = export_tflite_model(
        bump="patch",
        model_name="logistic_regression",
        config=cfg,
    )
    manifest2 = json.loads(result2.manifest_path.read_text(encoding="utf-8"))
    assert manifest2["latest_version"] == "1.0.1"
    assert set(manifest2["releases"]) >= {"1.0.0", "1.0.1"}
