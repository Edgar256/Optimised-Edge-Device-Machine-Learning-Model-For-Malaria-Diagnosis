"""Hyperparameter optimization for top-performing baseline models."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.evaluation.metrics import METRIC_NAMES, confidence_interval, cross_validate_model, cv_result_to_row
from src.models.registry import get_baseline_models
from src.models.train import _display_path, load_preprocessed_training_data
from src.utils.config import load_config
from src.utils.paths import resolve_path


@dataclass
class HyperparameterOptimizationResult:
    """Artifacts from hyperparameter optimization."""

    before: pd.DataFrame
    after: pd.DataFrame
    comparison: pd.DataFrame
    best_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _optimization_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("hyperparameter_optimization", {})


def _pythonify_param(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pythonify_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: _pythonify_param(value) for key, value in params.items()}


def _estimator_for_cross_validation(
    model_name: str,
    best_params: dict[str, Any],
    baseline_models: dict[str, Any],
) -> Any:
    """Return a clone-safe estimator for post-search cross-validation."""
    if model_name in {"catboost", "xgboost", "lightgbm"}:
        return baseline_models[model_name].set_params(**_pythonify_params(best_params))
    return baseline_models[model_name].set_params(**_pythonify_params(best_params))


def get_param_distributions(model_name: str) -> dict[str, Any]:
    """Return RandomizedSearchCV parameter distributions for a model."""
    if model_name == "logistic_regression":
        return {
            "model__C": loguniform(1e-3, 1e2),
            "model__l1_ratio": [0.0, 1.0],
            "model__solver": ["liblinear", "saga"],
            "model__class_weight": ["balanced", None],
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": randint(50, 250),
            "learning_rate": uniform(0.01, 0.19),
            "max_depth": randint(2, 6),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "subsample": uniform(0.7, 0.3),
        }
    if model_name == "decision_tree":
        return {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": randint(2, 30),
            "min_samples_leaf": randint(1, 15),
            "criterion": ["gini", "entropy", "log_loss"],
            "class_weight": ["balanced", None],
        }
    if model_name == "catboost":
        return {
            "depth": randint(3, 8),
            "learning_rate": uniform(0.01, 0.29),
            "iterations": randint(100, 400),
            "l2_leaf_reg": uniform(1.0, 9.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": randint(50, 300),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.29),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": randint(50, 300),
            "max_depth": randint(2, 8),
            "learning_rate": uniform(0.01, 0.29),
            "num_leaves": randint(15, 63),
            "subsample": uniform(0.6, 0.4),
        }
    raise ValueError(f"No hyperparameter distribution defined for model: {model_name}")


def is_optimizable_model(model_name: str) -> bool:
    """Return True when RandomizedSearchCV distributions exist for the model."""
    try:
        get_param_distributions(model_name)
    except ValueError:
        return False
    return True


def get_top_model_names(config: dict[str, Any] | None = None, top_n: int = 3) -> list[str]:
    """Read the top-ranked optimizable models from the baseline ranking table."""
    cfg = config or load_config()
    opt_cfg = _optimization_config(cfg)
    ranking_path = resolve_path(
        Path(cfg["paths"]["results_dir"]) / opt_cfg.get("ranking_source", "baseline_ranking.csv")
    )
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Baseline ranking not found at {ranking_path}. Run train-baselines first.")

    ranking = pd.read_csv(ranking_path)
    explicit = opt_cfg.get("model_names")
    if explicit:
        names = [str(name) for name in explicit]
        unsupported = [name for name in names if not is_optimizable_model(name)]
        if unsupported:
            raise ValueError(f"Models without hyperparameter search space: {unsupported}")
        return names

    selected: list[str] = []
    for model_name in ranking.sort_values("rank")["model_name"]:
        if not is_optimizable_model(model_name):
            continue
        selected.append(str(model_name))
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        raise ValueError(
            f"Only {len(selected)} optimizable models found in ranking; need {top_n}. "
            "Add param distributions or lower hyperparameter_optimization.top_n."
        )
    return selected


def _load_before_metrics(config: dict[str, Any], model_names: list[str]) -> pd.DataFrame:
    cfg = config
    opt_cfg = _optimization_config(cfg)
    comparison_path = resolve_path(
        Path(cfg["paths"]["results_dir"]) / opt_cfg.get("baseline_comparison_source", "baseline_comparison.csv")
    )
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Baseline comparison not found at {comparison_path}.")

    before = pd.read_csv(comparison_path)
    before = before[before["model_name"].isin(model_names)].copy()
    before.insert(0, "stage", "before_optimization")

    n_splits = int(cfg.get("training", {}).get("cv_folds", 5))
    for metric_name in METRIC_NAMES:
        if f"{metric_name}_ci_low" not in before.columns:
            ci_lows = []
            ci_highs = []
            for _, row in before.iterrows():
                low, high = confidence_interval(
                    row[f"{metric_name}_mean"],
                    row[f"{metric_name}_std"],
                    n_splits,
                )
                ci_lows.append(low)
                ci_highs.append(high)
            before[f"{metric_name}_ci_low"] = ci_lows
            before[f"{metric_name}_ci_high"] = ci_highs
    return before


def _build_delta_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in before["model_name"]:
        before_row = before.loc[before["model_name"] == model_name].iloc[0]
        after_row = after.loc[after["model_name"] == model_name].iloc[0]
        row: dict[str, Any] = {
            "model_name": model_name,
            "best_params": after_row.get("best_params", ""),
        }
        for metric_name in METRIC_NAMES:
            row[f"{metric_name}_before"] = before_row[f"{metric_name}_mean"]
            row[f"{metric_name}_after"] = after_row[f"{metric_name}_mean"]
            row[f"{metric_name}_delta"] = after_row[f"{metric_name}_mean"] - before_row[f"{metric_name}_mean"]
            row[f"{metric_name}_before_ci"] = (
                f"[{before_row[f'{metric_name}_ci_low']:.4f}, {before_row[f'{metric_name}_ci_high']:.4f}]"
            )
            row[f"{metric_name}_after_ci"] = (
                f"[{after_row[f'{metric_name}_ci_low']:.4f}, {after_row[f'{metric_name}_ci_high']:.4f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()]
    return "\n".join([headers, separator, *rows])


def optimize_top_models(config: dict[str, Any] | None = None) -> HyperparameterOptimizationResult:
    """Optimize hyperparameters for the top baseline models."""
    cfg = config or load_config()
    opt_cfg = _optimization_config(cfg)
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg["paths"]
    random_seed = int(cfg["project"].get("random_seed", 42))
    n_splits = int(train_cfg.get("cv_folds", 5))
    n_iter = int(opt_cfg.get("n_iter", 30))
    top_n = int(opt_cfg.get("top_n", 3))

    X, y, feature_names = load_preprocessed_training_data(cfg)
    model_names = get_top_model_names(cfg, top_n=top_n)
    baseline_models = get_baseline_models(random_seed=random_seed)

    results_dir = resolve_path(paths_cfg["results_dir"])
    models_dir = resolve_path(paths_cfg["models_dir"]) / "optimized"
    reports_dir = resolve_path(paths_cfg["reports_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    before_df = _load_before_metrics(cfg, model_names)
    after_rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for model_name in model_names:
        if model_name not in baseline_models:
            raise ValueError(f"Model {model_name!r} is not available in the baseline registry.")

        estimator = baseline_models[model_name]
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=get_param_distributions(model_name),
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=random_seed,
            n_jobs=-1,
            refit=True,
        )

        search_start = time.perf_counter()
        search.fit(X, y)
        search_time = time.perf_counter() - search_start

        optimized_estimator = search.best_estimator_
        cv_estimator = _estimator_for_cross_validation(model_name, search.best_params_, baseline_models)
        cv_result = cross_validate_model(
            cv_estimator,
            X,
            y,
            n_splits=n_splits,
            random_seed=random_seed,
        )
        best_params[model_name] = dict(search.best_params_)
        after_rows.append(
            cv_result_to_row(
                model_name,
                cv_result,
                stage="after_optimization",
                n_splits=n_splits,
                best_params=search.best_params_,
            )
        )
        after_rows[-1]["search_time_seconds"] = search_time
        after_rows[-1]["best_cv_score"] = float(search.best_score_)

        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(
            {
                "model_name": model_name,
                "estimator": optimized_estimator,
                "best_params": search.best_params_,
                "best_cv_score": float(search.best_score_),
                "feature_names": feature_names,
                "preprocessing_pipeline_path": _display_path(
                    resolve_path(
                        Path(paths_cfg["processed_data_dir"]) / cfg["preprocessing"]["pipeline_filename"]
                    )
                ),
            },
            model_path,
        )

    after_df = pd.DataFrame(after_rows)
    comparison_df = _build_delta_table(before_df, after_df)

    before_path = results_dir / opt_cfg.get("before_filename", "hyperparameter_before_optimization.csv")
    after_path = results_dir / opt_cfg.get("after_filename", "hyperparameter_after_optimization.csv")
    comparison_path = results_dir / opt_cfg.get("comparison_filename", "hyperparameter_comparison.csv")
    before_df.to_csv(before_path, index=False)
    after_df.to_csv(after_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_optimized": model_names,
        "cv_folds": n_splits,
        "n_iter": n_iter,
        "randomized_search_scoring": "roc_auc",
        "leakage_controls": [
            "Preprocessing pipeline is fitted upstream and only transforms features during tuning.",
            "RandomizedSearchCV operates on preprocessed matrices; labels never enter preprocessing.",
            "StratifiedKFold preserves class balance within each fold.",
        ],
        "before_path": _display_path(before_path),
        "after_path": _display_path(after_path),
        "comparison_path": _display_path(comparison_path),
        "optimized_models_dir": _display_path(models_dir),
        "best_params": best_params,
    }

    return HyperparameterOptimizationResult(
        before=before_df,
        after=after_df,
        comparison=comparison_df,
        best_params=best_params,
        stats=stats,
    )


def render_hyperparameter_report(result: HyperparameterOptimizationResult) -> str:
    """Render markdown report for hyperparameter optimization."""
    stats = result.stats
    display_cols = [
        "model_name",
        "roc_auc_mean",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "recall_mean",
        "recall_ci_low",
        "recall_ci_high",
        "f1_mean",
        "balanced_accuracy_mean",
    ]
    before_display = result.before[display_cols].copy()
    after_display = result.after[display_cols + ["best_params"]].copy()

    lines = [
        "# Hyperparameter Optimization Report",
        "",
        f"- **Generated (UTC):** {stats['generated_at_utc']}",
        f"- **Models optimized:** {', '.join(stats['models_optimized'])}",
        f"- **Search method:** RandomizedSearchCV (`n_iter={stats['n_iter']}`, scoring=`{stats['randomized_search_scoring']}`)",
        f"- **Cross-validation:** Stratified {stats['cv_folds']}-fold",
        "",
        "## Leakage prevention",
        "",
    ]
    for item in stats["leakage_controls"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Before optimization",
            "",
            _dataframe_to_markdown(before_display),
            "",
            "## After optimization",
            "",
            _dataframe_to_markdown(after_display),
            "",
            "## Best hyperparameters",
            "",
        ]
    )
    for model_name, params in stats["best_params"].items():
        lines.append(f"### `{model_name}`")
        lines.append("")
        lines.append(f"```json\n{json.dumps(params, indent=2)}\n```")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Before table: `{stats['before_path']}`",
            f"- After table: `{stats['after_path']}`",
            f"- Comparison table: `{stats['comparison_path']}`",
            f"- Optimized models: `{stats['optimized_models_dir']}/`",
            "",
        ]
    )
    return "\n".join(lines)


def write_hyperparameter_optimization_outputs(
    report_path: str | Path = "reports/hyperparameter_optimization_report.md",
    config: dict[str, Any] | None = None,
) -> HyperparameterOptimizationResult:
    """Run hyperparameter optimization and write all artifacts."""
    result = optimize_top_models(config=config)
    report_file = resolve_path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_hyperparameter_report(result), encoding="utf-8")
    result.stats["report_path"] = _display_path(report_file)
    return result
