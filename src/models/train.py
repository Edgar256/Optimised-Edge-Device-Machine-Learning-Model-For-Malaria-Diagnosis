"""Baseline model training orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from src.evaluation.metrics import cross_validate_model, measure_prediction_profile
from src.evaluation.plots import plot_confusion_matrix, plot_roc_comparison, plot_roc_curve
from src.evaluation.ranking import rank_models
from src.models.registry import get_baseline_models
from src.preprocessing.common import load_raw_dataframe
from src.utils.config import load_config
from src.utils.paths import resolve_path


@dataclass
class BaselineTrainingResult:
    """Artifacts from baseline model training."""

    comparison: pd.DataFrame
    ranking: pd.DataFrame
    trained_models: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("training", {})


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def load_preprocessed_training_data(config: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features via the fitted preprocessing pipeline and aligned labels."""
    cfg = config or load_config()
    paths_cfg = cfg["paths"]
    prep_cfg = cfg.get("preprocessing", {})

    pipeline_path = resolve_path(
        Path(paths_cfg["processed_data_dir"]) / prep_cfg.get("pipeline_filename", "preprocessing_pipeline.joblib")
    )
    processed_path = resolve_path(
        Path(paths_cfg["processed_data_dir"]) / prep_cfg.get("processed_dataset_filename", "processed_dataset.csv")
    )

    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}. Run preprocess first.")
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}. Run preprocess first.")

    raw_df, _ = load_raw_dataframe(cfg)
    preprocessing_pipeline = joblib.load(pipeline_path)
    processed_df = pd.read_csv(processed_path)

    X = preprocessing_pipeline.transform(raw_df)
    y = processed_df["target_binary"].to_numpy()
    feature_names = preprocessing_pipeline.named_steps["column_transformer"].get_feature_names_out().tolist()

    if len(X) != len(y):
        raise ValueError(f"Feature rows ({len(X)}) and label rows ({len(y)}) are misaligned.")

    return np.asarray(X), np.asarray(y), feature_names


def _model_artifact_path(models_dir: Path, model_name: str) -> Path:
    return models_dir / "baseline" / f"{model_name}.joblib"


def _serialize_model_size_mb(model_path: Path) -> float:
    return round(model_path.stat().st_size / (1024 * 1024), 4)


def train_baseline_models(config: dict[str, Any] | None = None) -> BaselineTrainingResult:
    """Train and evaluate all baseline models using stratified cross-validation."""
    cfg = config or load_config()
    train_cfg = _training_config(cfg)
    paths_cfg = cfg["paths"]
    random_seed = int(cfg["project"].get("random_seed", 42))
    n_splits = int(train_cfg.get("cv_folds", 5))

    X, y, feature_names = load_preprocessed_training_data(cfg)
    models = get_baseline_models(random_seed=random_seed)

    models_dir = resolve_path(paths_cfg["models_dir"])
    results_dir = resolve_path(paths_cfg["results_dir"])
    figures_dir = resolve_path(paths_cfg["figures_dir"])
    reports_dir = resolve_path(paths_cfg["reports_dir"])
    baseline_models_dir = models_dir / "baseline"
    confusion_dir = figures_dir / "confusion_matrices"
    roc_dir = figures_dir / "roc_curves"
    for directory in (baseline_models_dir, results_dir, confusion_dir, roc_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name, estimator in models.items():
        cv_result = cross_validate_model(
            estimator,
            X,
            y,
            n_splits=n_splits,
            random_seed=random_seed,
        )
        fitted_model = clone(estimator).fit(X, y)
        model_path = _model_artifact_path(models_dir, model_name)
        joblib.dump(
            {
                "model_name": model_name,
                "estimator": fitted_model,
                "feature_names": feature_names,
                "preprocessing_pipeline_path": _display_path(
                    resolve_path(Path(paths_cfg["processed_data_dir"]) / cfg["preprocessing"]["pipeline_filename"])
                ),
            },
            model_path,
        )
        trained_models[model_name] = fitted_model
        inference_profile = measure_prediction_profile(fitted_model, X)

        row = {
            "model_name": model_name,
            "accuracy_mean": cv_result["accuracy_mean"],
            "accuracy_std": cv_result["accuracy_std"],
            "precision_mean": cv_result["precision_mean"],
            "precision_std": cv_result["precision_std"],
            "recall_mean": cv_result["recall_mean"],
            "recall_std": cv_result["recall_std"],
            "specificity_mean": cv_result["specificity_mean"],
            "specificity_std": cv_result["specificity_std"],
            "sensitivity_mean": cv_result["sensitivity_mean"],
            "sensitivity_std": cv_result["sensitivity_std"],
            "roc_auc_mean": cv_result["roc_auc_mean"],
            "roc_auc_std": cv_result["roc_auc_std"],
            "f1_mean": cv_result["f1_mean"],
            "f1_std": cv_result["f1_std"],
            "balanced_accuracy_mean": cv_result["balanced_accuracy_mean"],
            "balanced_accuracy_std": cv_result["balanced_accuracy_std"],
            "training_time_seconds_mean": cv_result["training_time_seconds_mean"],
            "training_time_seconds_std": cv_result["training_time_seconds_std"],
            "prediction_time_seconds_mean": cv_result["prediction_time_seconds_mean"],
            "prediction_time_seconds_std": cv_result["prediction_time_seconds_std"],
            "model_size_mb": _serialize_model_size_mb(model_path),
            "memory_usage_mb": round(inference_profile["memory_usage_mb"], 4),
        }
        rows.append(row)

        plot_confusion_matrix(
            cv_result["confusion_matrix_sum"],
            confusion_dir / f"{model_name}.png",
            title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        )
        if cv_result["y_proba"] is not None:
            roc_curves[model_name] = (cv_result["y_true"], cv_result["y_proba"])
            plot_roc_curve(
                cv_result["y_true"],
                cv_result["y_proba"],
                roc_dir / f"{model_name}.png",
                title=f"{model_name.replace('_', ' ').title()} ROC",
            )

    comparison = pd.DataFrame(rows)
    ranking = rank_models(comparison)

    comparison_path = results_dir / train_cfg.get("comparison_filename", "baseline_comparison.csv")
    ranking_path = results_dir / train_cfg.get("ranking_filename", "baseline_ranking.csv")
    comparison.to_csv(comparison_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    if roc_curves:
        plot_roc_comparison(roc_curves, figures_dir / "roc_comparison.png")

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_trained": list(models.keys()),
        "model_count": len(models),
        "samples": len(y),
        "feature_count": X.shape[1],
        "cv_folds": n_splits,
        "comparison_path": _display_path(comparison_path),
        "ranking_path": _display_path(ranking_path),
        "best_model": ranking.loc[0, "model_name"],
        "best_rank_score": float(ranking.loc[0, "rank_score"]),
    }
    manifest_path = results_dir / train_cfg.get("manifest_filename", "baseline_training_manifest.json")
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    stats["manifest_path"] = _display_path(manifest_path)

    return BaselineTrainingResult(
        comparison=comparison,
        ranking=ranking,
        trained_models=trained_models,
        stats=stats,
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()]
    return "\n".join([headers, separator, *rows])


def render_baseline_training_report(result: BaselineTrainingResult) -> str:
    """Render a markdown summary of baseline training."""
    stats = result.stats
    lines = [
        "# Baseline Model Training Report",
        "",
        f"- **Generated (UTC):** {stats.get('generated_at_utc')}",
        f"- **Samples:** {stats.get('samples')}",
        f"- **Features:** {stats.get('feature_count')} (from fitted preprocessing pipeline)",
        f"- **CV folds:** {stats.get('cv_folds')} (stratified)",
        f"- **Models trained:** {stats.get('model_count')}",
        f"- **Best model:** `{stats.get('best_model')}` (rank score {stats.get('best_rank_score'):.4f})",
        "",
        "## Model ranking",
        "",
        _dataframe_to_markdown(result.ranking),
        "",
        "## Artifacts",
        "",
        f"- Comparison table: `{stats.get('comparison_path')}`",
        f"- Ranking table: `{stats.get('ranking_path')}`",
        f"- Manifest: `{stats.get('manifest_path')}`",
        "- Confusion matrices: `figures/confusion_matrices/`",
        "- ROC curves: `figures/roc_curves/`",
        "- Combined ROC plot: `figures/roc_comparison.png`",
        "- Saved models: `models/baseline/`",
        "",
    ]
    return "\n".join(lines)


def write_baseline_training_outputs(
    report_path: str | Path = "reports/baseline_training_report.md",
    config: dict[str, Any] | None = None,
) -> BaselineTrainingResult:
    """Train baselines and write all evaluation artifacts."""
    result = train_baseline_models(config=config)
    report_file = resolve_path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_baseline_training_report(result), encoding="utf-8")
    result.stats["report_path"] = _display_path(report_file)
    return result
