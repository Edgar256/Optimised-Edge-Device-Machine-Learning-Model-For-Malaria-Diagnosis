"""Runtime state shared across API modules."""

from __future__ import annotations

from src.deployment.predictor import PredictorArtifacts

_artifacts: PredictorArtifacts | None = None


def get_predictor_artifacts() -> PredictorArtifacts | None:
    return _artifacts


def set_predictor_artifacts(artifacts: PredictorArtifacts | None) -> None:
    global _artifacts
    _artifacts = artifacts
