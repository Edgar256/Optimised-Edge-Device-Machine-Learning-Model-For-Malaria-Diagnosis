"""Model ranking utilities."""

from __future__ import annotations

import pandas as pd


RANKING_COLUMNS = [
    "rank",
    "model_name",
    "rank_score",
    "roc_auc_mean",
    "recall_mean",
    "f1_mean",
    "balanced_accuracy_mean",
    "accuracy_mean",
    "precision_mean",
    "specificity_mean",
    "sensitivity_mean",
    "training_time_seconds_mean",
    "prediction_time_seconds_mean",
    "model_size_mb",
    "memory_usage_mb",
]


def rank_models(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models using a weighted clinical scoring function."""
    ranked = comparison_df.copy()
    train_time = ranked["training_time_seconds_mean"].fillna(ranked["training_time_seconds_mean"].max())
    max_train = train_time.max() if train_time.max() > 0 else 1.0
    train_norm = 1.0 - (train_time / max_train)

    ranked["rank_score"] = (
        0.35 * ranked["roc_auc_mean"].fillna(0)
        + 0.25 * ranked["recall_mean"].fillna(0)
        + 0.15 * ranked["f1_mean"].fillna(0)
        + 0.15 * ranked["balanced_accuracy_mean"].fillna(0)
        + 0.10 * train_norm
    )
    ranked = ranked.sort_values(
        by=["rank_score", "roc_auc_mean", "recall_mean", "model_size_mb"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked[RANKING_COLUMNS]
