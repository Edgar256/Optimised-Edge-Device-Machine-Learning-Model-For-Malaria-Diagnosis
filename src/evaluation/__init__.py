"""Offline metrics, cross-validation, and edge-device performance analysis."""

from src.evaluation.metrics import (
    METRIC_NAMES,
    confidence_interval,
    compute_classification_metrics,
    cross_validate_model,
    cv_result_to_row,
)
from src.evaluation.ranking import rank_models

__all__ = [
    "METRIC_NAMES",
    "compute_classification_metrics",
    "confidence_interval",
    "cross_validate_model",
    "cv_result_to_row",
    "rank_models",
]
