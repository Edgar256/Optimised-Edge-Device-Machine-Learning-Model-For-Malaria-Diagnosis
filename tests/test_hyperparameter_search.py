"""Tests for hyperparameter optimization."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import confidence_interval, cv_result_to_row
from src.models.hyperparameter_search import get_param_distributions, get_top_model_names


def test_confidence_interval_narrow_with_more_folds() -> None:
    low, high = confidence_interval(0.8, 0.1, n=5)
    assert low < 0.8 < high


def test_get_param_distributions_for_top_models() -> None:
    for model_name in ("logistic_regression", "gradient_boosting", "decision_tree", "catboost"):
        params = get_param_distributions(model_name)
        assert len(params) >= 3


def test_logistic_regression_search_space_has_no_elasticnet() -> None:
    params = get_param_distributions("logistic_regression")
    assert "model__penalty" not in params
    assert "model__l1_ratio" in params


def test_get_top_model_names_from_ranking() -> None:
    try:
        names = get_top_model_names(top_n=3)
    except FileNotFoundError:
        pytest.skip("baseline ranking not found")
    assert len(names) == 3


def test_cv_result_to_row_includes_ci() -> None:
    row = cv_result_to_row(
        "logistic_regression",
        {
            "accuracy_mean": 0.9,
            "accuracy_std": 0.02,
            "precision_mean": 0.8,
            "precision_std": 0.03,
            "recall_mean": 0.7,
            "recall_std": 0.04,
            "specificity_mean": 0.95,
            "specificity_std": 0.01,
            "sensitivity_mean": 0.7,
            "sensitivity_std": 0.04,
            "roc_auc_mean": 0.88,
            "roc_auc_std": 0.03,
            "f1_mean": 0.75,
            "f1_std": 0.03,
            "balanced_accuracy_mean": 0.82,
            "balanced_accuracy_std": 0.02,
            "training_time_seconds_mean": 0.1,
            "prediction_time_seconds_mean": 0.01,
        },
        stage="after_optimization",
        n_splits=5,
        best_params={"model__C": 1.0},
    )
    assert "roc_auc_ci_low" in row
    assert row["best_params"] == "{'model__C': 1.0}"
