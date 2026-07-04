"""Model explainability analysis and publication figures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from src.models.train import _display_path, load_preprocessed_training_data
from src.utils.config import load_config
from src.utils.paths import resolve_path

SYMPTOM_PREFIXES = [
    "Fever (Yes/No)",
    "Headache (Yes/No)",
    "Chills (Yes/No)",
    "Vomiting (Yes/No)",
    "Fatigue (Yes/No)",
    "Anemia Signs (Yes/No)",
    "Other Symptoms (Specify)",
]

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@dataclass
class ExplainabilityResult:
    """Container for explainability artifacts."""

    model_name: str
    feature_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    symptom_ranking: pd.DataFrame
    shap_summary: pd.DataFrame | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def _explainability_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("explainability", {})


def resolve_best_model_name(config: dict[str, Any] | None = None) -> str:
    """Auto-select the best model from the latest evaluation results.

    Priority:
    1. Rank 1 in ``baseline_ranking.csv`` (composite clinical / edge score)
    2. Highest ``roc_auc_mean`` in ``hyperparameter_after_optimization.csv``

    The selected name is then loaded from ``models/optimized/`` when available,
    otherwise ``models/baseline/``.

    Optional manual overrides live in config (``api.model_name`` /
    ``explainability.model_name``) and are applied by callers, not here.
    """
    cfg = config or load_config()

    ranking_path = resolve_path(
        Path(cfg["paths"]["results_dir"])
        / cfg.get("training", {}).get("ranking_filename", "baseline_ranking.csv")
    )
    if ranking_path.is_file():
        ranking = pd.read_csv(ranking_path)
        if not ranking.empty and "rank" in ranking.columns:
            return str(ranking.sort_values("rank").iloc[0]["model_name"])

    after_path = resolve_path(
        Path(cfg["paths"]["results_dir"])
        / cfg.get("hyperparameter_optimization", {}).get(
            "after_filename", "hyperparameter_after_optimization.csv"
        )
    )
    if after_path.is_file():
        after = pd.read_csv(after_path)
        if not after.empty and "roc_auc_mean" in after.columns:
            return str(after.sort_values("roc_auc_mean", ascending=False).iloc[0]["model_name"])

    raise FileNotFoundError(
        "No baseline ranking or optimization results found. "
        "Run train-baselines (and preferably optimize-models) first."
    )


def _optional_model_override(value: Any) -> str | None:
    """Return a non-empty model name override, or None for auto-select."""
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def load_model_artifact(model_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load an optimized model artifact, falling back to baseline if needed."""
    cfg = config or load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    optimized_path = models_dir / "optimized" / f"{model_name}.joblib"
    baseline_path = models_dir / "baseline" / f"{model_name}.joblib"

    if optimized_path.is_file():
        return joblib.load(optimized_path)
    if baseline_path.is_file():
        return joblib.load(baseline_path)
    raise FileNotFoundError(f"No saved model found for {model_name!r}.")


def _get_estimator(artifact: dict[str, Any]) -> Any:
    return artifact["estimator"]


def _apply_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(PUBLICATION_RC)


def _feature_importance_table(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        inner = getattr(model, "named_steps", {}).get("model", model) if hasattr(model, "named_steps") else model
        if not hasattr(inner, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance", "rank"])
        importances = inner.feature_importances_
    else:
        importances = model.feature_importances_

    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def _permutation_importance_table(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    random_seed: int,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=20,
        random_state=random_seed,
        scoring="roc_auc",
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    table = table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def _symptom_group_name(feature_name: str) -> str | None:
    for prefix in SYMPTOM_PREFIXES:
        if feature_name.startswith(prefix):
            return prefix
    return None


def _aggregate_symptom_importance(importance_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {prefix: [] for prefix in SYMPTOM_PREFIXES}

    for _, row in importance_df.iterrows():
        group = _symptom_group_name(str(row["feature"]))
        if group is not None:
            grouped[group].append(float(row[value_column]))

    for symptom, values in grouped.items():
        if values:
            rows.append(
                {
                    "symptom": symptom,
                    "aggregate_importance": float(np.sum(values)),
                    "mean_importance": float(np.mean(values)),
                    "feature_count": len(values),
                }
            )

    table = pd.DataFrame(rows).sort_values("aggregate_importance", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def _compute_shap_summary(model: Any, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame | None:
    try:
        import shap
    except ImportError:
        return None

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def _plot_horizontal_importance(df: pd.DataFrame, value_col: str, error_col: str | None, title: str, path: Path) -> None:
    _apply_publication_style()
    top = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    if error_col and error_col in top.columns:
        ax.barh(top["feature"], top[value_col], xerr=top[error_col], color="#2C6E8C", alpha=0.9)
    else:
        ax.barh(top["feature"], top[value_col], color="#2C6E8C", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_symptom_ranking(df: pd.DataFrame, title: str, path: Path) -> None:
    _apply_publication_style()
    top = df.sort_values("aggregate_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["symptom"], top["aggregate_importance"], color="#A23B72", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Aggregate importance")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_figures(model: Any, X: np.ndarray, feature_names: list[str], output_dir: Path) -> bool:
    try:
        import shap
    except ImportError:
        return False

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    _apply_publication_style()
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", bbox_inches="tight", dpi=300)
    plt.close()
    return True


def _plot_partial_dependence(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    features: list[str],
    output_dir: Path,
) -> list[str]:
    generated: list[str] = []
    feature_indices = [feature_names.index(name) for name in features if name in feature_names]
    if not feature_indices:
        return generated

    _apply_publication_style()
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=feature_indices,
        feature_names=feature_names,
        grid_resolution=40,
    )
    fig = display.figure_
    fig.suptitle("Partial Dependence Plots", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "partial_dependence_panels.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    generated.append("partial_dependence_panels.png")

    for index in feature_indices:
        single_display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[index],
            feature_names=feature_names,
            grid_resolution=40,
        )
        fig_single = single_display.figure_
        feature_label = feature_names[index].replace("/", "_").replace(" ", "_")
        filename = f"pdp_{feature_label}.png"
        fig_single.tight_layout()
        fig_single.savefig(output_dir / filename, bbox_inches="tight", dpi=300)
        plt.close(fig_single)
        generated.append(filename)
    return generated


def _auto_symptom_narrative(symptom_ranking: pd.DataFrame, source: str) -> str:
    if symptom_ranking.empty:
        return f"No symptom-level drivers were identified from {source}."

    top = symptom_ranking.head(3)
    lines = [
        f"Based on **{source}**, the symptoms most associated with malaria diagnosis are:",
        "",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"- **{row['symptom']}** (aggregate score {row['aggregate_importance']:.4f})"
        )
    lines.append("")
    lines.append(
        "Positive associations for binary symptom features indicate that reporting the symptom "
        "increases predicted malaria risk relative to the reference encoded category."
    )
    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()]
    return "\n".join([headers, separator, *rows])


def render_explainability_report(result: ExplainabilityResult) -> str:
    """Render markdown explainability report."""
    stats = result.stats
    lines = [
        "# Best Model Explainability Report",
        "",
        f"- **Model:** `{result.model_name}`",
        f"- **Generated (UTC):** {stats.get('generated_at_utc')}",
        f"- **Samples analysed:** {stats.get('samples')}",
        f"- **Features:** {stats.get('feature_count')}",
        "",
        "## Model performance context",
        "",
        stats.get("performance_summary", ""),
        "",
        "## Symptom influence summary",
        "",
        stats.get("symptom_narrative_permutation", ""),
        "",
        stats.get("symptom_narrative_shap", ""),
        "",
        "## Top overall features (permutation importance)",
        "",
    ]
    top_perm = result.permutation_importance.head(10)[
        ["rank", "feature", "importance_mean", "importance_std"]
    ]
    lines.append(_dataframe_to_markdown(top_perm))
    lines.extend(
        [
            "",
            "## Symptom ranking (aggregated permutation importance)",
            "",
            _dataframe_to_markdown(result.symptom_ranking),
            "",
            "## Artifacts",
            "",
            f"- Figures: `{stats.get('figures_dir')}`",
            f"- Tables: `{stats.get('tables_dir')}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_explainability(config: dict[str, Any] | None = None) -> ExplainabilityResult:
    """Generate explainability artifacts for the best-performing model."""
    cfg = config or load_config()
    exp_cfg = _explainability_config(cfg)
    random_seed = int(cfg["project"].get("random_seed", 42))

    model_name = _optional_model_override(exp_cfg.get("model_name")) or resolve_best_model_name(cfg)
    artifact = load_model_artifact(model_name, cfg)
    model = _get_estimator(artifact)
    X, y, feature_names = load_preprocessed_training_data(cfg)
    if artifact.get("feature_names"):
        feature_names = list(artifact["feature_names"])

    output_dir = resolve_path(exp_cfg.get("output_dir", "reports/explainability"))
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    feature_importance = _feature_importance_table(model, feature_names)
    perm_importance = _permutation_importance_table(model, X, y, feature_names, random_seed)
    symptom_ranking = _aggregate_symptom_importance(perm_importance, "importance_mean")
    shap_summary = _compute_shap_summary(model, X, feature_names)

    feature_importance.to_csv(tables_dir / "feature_importance.csv", index=False)
    perm_importance.to_csv(tables_dir / "permutation_importance.csv", index=False)
    symptom_ranking.to_csv(tables_dir / "symptom_ranking.csv", index=False)
    if shap_summary is not None:
        shap_summary.to_csv(tables_dir / "shap_mean_abs.csv", index=False)
        symptom_shap = _aggregate_symptom_importance(
            shap_summary.rename(columns={"mean_abs_shap": "importance_mean"}),
            "importance_mean",
        )
        symptom_shap.to_csv(tables_dir / "symptom_ranking_shap.csv", index=False)
    else:
        symptom_shap = pd.DataFrame()

    _plot_horizontal_importance(
        feature_importance,
        "importance",
        None,
        f"{model_name.replace('_', ' ').title()} Feature Importance",
        figures_dir / "feature_importance.png",
    )
    _plot_horizontal_importance(
        perm_importance,
        "importance_mean",
        "importance_std",
        f"{model_name.replace('_', ' ').title()} Permutation Importance",
        figures_dir / "permutation_importance.png",
    )
    _plot_symptom_ranking(
        symptom_ranking,
        "Symptom Influence on Malaria Diagnosis",
        figures_dir / "symptom_influence_permutation.png",
    )
    if not symptom_shap.empty:
        _plot_symptom_ranking(
            symptom_shap,
            "Symptom Influence (Mean |SHAP|)",
            figures_dir / "symptom_influence_shap.png",
        )

    shap_generated = _plot_shap_figures(model, X, feature_names, figures_dir)
    pdp_features = [
        name
        for name in [
            "Age",
            "Fever Duration (Days)",
            "Fever (Yes/No)_yes",
            "Headache (Yes/No)_yes",
            "Vomiting (Yes/No)_yes",
        ]
        if name in feature_names
    ]
    pdp_files = _plot_partial_dependence(model, X, feature_names, pdp_features, figures_dir)

    after_path = resolve_path(
        Path(cfg["paths"]["results_dir"])
        / cfg.get("hyperparameter_optimization", {}).get("after_filename", "hyperparameter_after_optimization.csv")
    )
    performance_summary = ""
    if after_path.is_file():
        row = pd.read_csv(after_path)
        row = row[row["model_name"] == model_name].iloc[0]
        performance_summary = (
            f"The selected model achieved ROC AUC **{row['roc_auc_mean']:.3f}** "
            f"(95% CI [{row['roc_auc_ci_low']:.3f}, {row['roc_auc_ci_high']:.3f}]), "
            f"recall **{row['recall_mean']:.3f}**, and F1 **{row['f1_mean']:.3f}** after hyperparameter optimization."
        )

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "samples": len(y),
        "feature_count": len(feature_names),
        "figures_dir": _display_path(figures_dir),
        "tables_dir": _display_path(tables_dir),
        "performance_summary": performance_summary,
        "symptom_narrative_permutation": _auto_symptom_narrative(symptom_ranking, "permutation importance"),
        "symptom_narrative_shap": _auto_symptom_narrative(symptom_shap, "SHAP values")
        if not symptom_shap.empty
        else "SHAP analysis was skipped because the `shap` package is not installed.",
        "shap_generated": shap_generated,
        "pdp_files": pdp_files,
        "best_params": artifact.get("best_params"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    return ExplainabilityResult(
        model_name=model_name,
        feature_importance=feature_importance,
        permutation_importance=perm_importance,
        symptom_ranking=symptom_ranking,
        shap_summary=shap_summary,
        stats=stats,
    )


def write_explainability_outputs(
    report_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> ExplainabilityResult:
    """Run explainability analysis and write report artifacts."""
    cfg = config or load_config()
    exp_cfg = _explainability_config(cfg)
    output_dir = resolve_path(exp_cfg.get("output_dir", "reports/explainability"))
    report_file = resolve_path(report_path or output_dir / "best_model_explainability.md")

    result = run_explainability(cfg)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_explainability_report(result), encoding="utf-8")
    result.stats["report_path"] = _display_path(report_file)
    return result
