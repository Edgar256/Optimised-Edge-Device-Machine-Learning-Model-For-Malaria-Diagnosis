"""Classification metrics for model evaluation."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "sensitivity",
    "roc_auc",
    "f1",
    "balanced_accuracy",
]


def confidence_interval(mean: float, std: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute a two-sided confidence interval from CV fold statistics."""
    if n <= 1 or np.isnan(mean):
        return float("nan"), float("nan")
    from scipy import stats

    alpha = 1.0 - confidence
    t_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    margin = t_critical * (std / np.sqrt(n))
    return float(mean - margin), float(mean + margin)


def cv_result_to_row(
    model_name: str,
    cv_result: dict[str, Any],
    *,
    stage: str,
    n_splits: int,
    best_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten cross-validation results into a comparison-table row with CIs."""
    row: dict[str, Any] = {
        "model_name": model_name,
        "stage": stage,
    }
    if best_params is not None:
        row["best_params"] = str(best_params)

    for metric_name in METRIC_NAMES:
        mean = cv_result[f"{metric_name}_mean"]
        std = cv_result[f"{metric_name}_std"]
        ci_low, ci_high = confidence_interval(mean, std, n_splits)
        row[f"{metric_name}_mean"] = mean
        row[f"{metric_name}_std"] = std
        row[f"{metric_name}_ci_low"] = ci_low
        row[f"{metric_name}_ci_high"] = ci_high

    row["training_time_seconds_mean"] = cv_result["training_time_seconds_mean"]
    row["prediction_time_seconds_mean"] = cv_result["prediction_time_seconds_mean"]
    return row


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    return tn / denominator


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sensitivity (true positive rate), equivalent to recall for the positive class."""
    return float(recall_score(y_true, y_pred, zero_division=0))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_score(y_true, y_pred)),
        "sensitivity": float(sensitivity_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def measure_prediction_profile(model: Any, X: np.ndarray) -> dict[str, float]:
    """Measure prediction latency and peak memory during inference."""
    sample = X[: min(len(X), 64)]
    tracemalloc.start()
    start = time.perf_counter()
    model.predict(sample)
    if hasattr(model, "predict_proba"):
        model.predict_proba(sample)
    prediction_time_seconds = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "prediction_time_seconds": prediction_time_seconds,
        "memory_usage_mb": peak_memory / (1024 * 1024),
    }


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run stratified k-fold cross-validation and aggregate metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics: list[dict[str, float]] = []
    fold_train_times: list[float] = []
    fold_pred_times: list[float] = []
    confusion_matrices: list[np.ndarray] = []
    y_true_all: list[int] = []
    y_proba_all: list[float] = []

    for train_index, test_index in cv.split(X, y):
        estimator = clone(model)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fold_train_times.append(time.perf_counter() - train_start)

        pred_start = time.perf_counter()
        y_pred = estimator.predict(X_test)
        fold_pred_times.append(time.perf_counter() - pred_start)

        y_proba = estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
        fold_metrics.append(compute_classification_metrics(y_test, y_pred, y_proba))
        confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        y_true_all.extend(y_test.tolist())
        if y_proba is not None:
            y_proba_all.extend(y_proba.tolist())

    metric_names = METRIC_NAMES
    summary: dict[str, Any] = {
        "fold_metrics": fold_metrics,
        "confusion_matrix_sum": sum(confusion_matrices),
        "y_true": np.asarray(y_true_all),
        "y_proba": np.asarray(y_proba_all) if y_proba_all else None,
        "training_time_seconds_mean": float(np.mean(fold_train_times)),
        "training_time_seconds_std": float(np.std(fold_train_times)),
        "prediction_time_seconds_mean": float(np.mean(fold_pred_times)),
        "prediction_time_seconds_std": float(np.std(fold_pred_times)),
    }
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        summary[f"{metric_name}_mean"] = float(np.nanmean(values))
        summary[f"{metric_name}_std"] = float(np.nanstd(values))
    return summary
