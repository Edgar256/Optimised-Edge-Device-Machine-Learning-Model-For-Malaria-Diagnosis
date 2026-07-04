"""Tests for model explainability."""

import pandas as pd
import pytest

from src.evaluation.explainability import (
    _aggregate_symptom_importance,
    _auto_symptom_narrative,
    _symptom_group_name,
    resolve_best_model_name,
)


def test_symptom_group_name_maps_encoded_features() -> None:
    assert _symptom_group_name("Fever (Yes/No)_yes") == "Fever (Yes/No)"
    assert _symptom_group_name("Age") is None


def test_aggregate_symptom_importance_sums_related_features() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["Fever (Yes/No)_yes", "Fever (Yes/No)_no", "Age"],
            "importance_mean": [0.05, 0.01, 0.02],
        }
    )
    aggregated = _aggregate_symptom_importance(importance, "importance_mean")
    fever_row = aggregated.loc[aggregated["symptom"] == "Fever (Yes/No)"].iloc[0]
    assert fever_row["aggregate_importance"] == pytest.approx(0.06)


def test_auto_symptom_narrative_mentions_top_symptoms() -> None:
    ranking = pd.DataFrame(
        {
            "symptom": ["Fever (Yes/No)", "Headache (Yes/No)"],
            "aggregate_importance": [0.08, 0.04],
            "mean_importance": [0.04, 0.04],
            "feature_count": [2, 2],
        }
    )
    narrative = _auto_symptom_narrative(ranking, "permutation importance")
    assert "Fever (Yes/No)" in narrative


def test_resolve_best_model_name_auto_selects_by_baseline_rank() -> None:
    from src.utils.paths import resolve_path

    try:
        name = resolve_best_model_name()
    except FileNotFoundError:
        pytest.skip("ranking or optimization results not found")

    ranking_path = resolve_path("results/baseline_ranking.csv")
    if ranking_path.is_file():
        ranking = pd.read_csv(ranking_path).sort_values("rank")
        assert name == ranking.iloc[0]["model_name"]
    else:
        after = resolve_path("results/hyperparameter_after_optimization.csv")
        best = pd.read_csv(after).sort_values("roc_auc_mean", ascending=False)
        assert name == best.iloc[0]["model_name"]
