"""Offline metrics, cross-validation, and edge-device performance analysis."""

from src.evaluation.chapter4_tables import write_chapter4_tables
from src.evaluation.explainability import write_explainability_outputs
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
    "write_chapter4_tables",
    "write_explainability_outputs",
    "rank_models",
]
