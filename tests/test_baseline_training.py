"""Tests for baseline model training."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_classification_metrics, cross_validate_model, specificity_score
from src.evaluation.ranking import rank_models
from src.models.registry import get_baseline_models
from src.models.train import load_preprocessed_training_data


def test_specificity_and_sensitivity() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5


def test_registry_includes_core_models() -> None:
    models = get_baseline_models(random_seed=42)
    for name in [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "support_vector_machine",
        "mlp",
    ]:
        assert name in models


def test_cross_validate_model_returns_metrics() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        weights=[0.8, 0.2],
        random_state=42,
    )
    result = cross_validate_model(LogisticRegression(max_iter=1000), X, y, n_splits=3, random_seed=42)
    assert "roc_auc_mean" in result
    assert result["accuracy_mean"] >= 0.0


def test_rank_models_orders_by_score() -> None:
    import pandas as pd

    comparison = pd.DataFrame(
        [
            {"model_name": "a", "roc_auc_mean": 0.7, "recall_mean": 0.5, "f1_mean": 0.5, "balanced_accuracy_mean": 0.6, "accuracy_mean": 0.8, "precision_mean": 0.5, "specificity_mean": 0.8, "sensitivity_mean": 0.5, "training_time_seconds_mean": 1.0, "prediction_time_seconds_mean": 0.01, "model_size_mb": 1.0, "memory_usage_mb": 1.0},
            {"model_name": "b", "roc_auc_mean": 0.9, "recall_mean": 0.8, "f1_mean": 0.7, "balanced_accuracy_mean": 0.8, "accuracy_mean": 0.85, "precision_mean": 0.7, "specificity_mean": 0.85, "sensitivity_mean": 0.8, "training_time_seconds_mean": 1.0, "prediction_time_seconds_mean": 0.01, "model_size_mb": 1.0, "memory_usage_mb": 1.0},
        ]
    )
    ranked = rank_models(comparison)
    assert ranked.loc[0, "model_name"] == "b"


def test_load_preprocessed_training_data_shape() -> None:
    try:
        X, y, feature_names = load_preprocessed_training_data()
    except FileNotFoundError:
        pytest.skip("preprocessing artifacts not found")
    assert len(X) == len(y)
    assert len(feature_names) == X.shape[1]
