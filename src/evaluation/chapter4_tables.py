"""Generate Chapter 4 thesis tables and publication-quality figure exports."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.explainability import load_model_artifact, resolve_best_model_name
from src.evaluation.metrics import METRIC_NAMES, measure_prediction_profile
from src.models.train import _serialize_model_size_mb, load_preprocessed_training_data
from src.preprocessing.audit import run_audit
from src.preprocessing.common import is_missing, load_raw_dataframe, model_feature_columns
from src.utils.config import load_config
from src.utils.paths import resolve_path

PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
}

TABLE_SPECS: dict[str, dict[str, str]] = {
    "table_4_1_dataset_summary": {
        "title": "Table 4.1 Dataset Summary",
        "caption": (
            "Table 4.1 Summary of the Hoima District clinical dataset used for malaria diagnosis "
            "modelling, including record counts, class distribution, and preprocessing outcomes."
        ),
    },
    "table_4_2_missing_values": {
        "title": "Table 4.2 Missing Values",
        "caption": (
            "Table 4.2 Missing-value profile for clinical intake variables in the raw Kobo export "
            "before preprocessing (n = total raw records)."
        ),
    },
    "table_4_3_feature_statistics": {
        "title": "Table 4.3 Feature Statistics",
        "caption": (
            "Table 4.3 Descriptive statistics for modelled clinical features after preprocessing "
            "and prior to one-hot encoding."
        ),
    },
    "table_4_4_baseline_results": {
        "title": "Table 4.4 Baseline Results",
        "caption": (
            "Table 4.4 Stratified five-fold cross-validation performance of baseline classifiers "
            "on the preprocessed feature matrix (mean ± standard deviation)."
        ),
    },
    "table_4_5_optimized_results": {
        "title": "Table 4.5 Optimized Results",
        "caption": (
            "Table 4.5 Performance of hyperparameter-tuned models after RandomizedSearchCV, "
            "reported with 95% confidence intervals from cross-validation folds."
        ),
    },
    "table_4_6_model_comparison": {
        "title": "Table 4.6 Model Comparison",
        "caption": (
            "Table 4.6 Before-and-after comparison of key metrics for tuned models, showing "
            "absolute change (Δ) following hyperparameter optimization."
        ),
    },
    "table_4_7_edge_performance": {
        "title": "Table 4.7 Edge Performance",
        "caption": (
            "Table 4.7 Edge-deployment resource profile: serialized model size, cross-validated "
            "inference latency, peak memory during batch prediction, and single-patient API latency."
        ),
    },
}


@dataclass
class Chapter4TablesResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    captions: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def _chapter4_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("chapter4", {})


def _format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _format_ci(mean: float, ci_low: float, ci_high: float, decimals: int = 3) -> str:
    if pd.isna(mean):
        return "—"
    return f"{mean:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def build_table_4_1_dataset_summary(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    data_cfg = cfg["data"]
    audit = run_audit(cfg)
    findings = audit.findings
    target_column = data_cfg["target_column"]
    positive = data_cfg["target_positive_label"]
    negative = data_cfg["target_negative_label"]

    target_dist = findings["categorical_checks"]["target_distribution"]
    malaria_count = int(target_dist.get(positive, 0))
    not_malaria_count = int(target_dist.get(negative, 0))
    labelled = malaria_count + not_malaria_count
    raw_rows = findings["shape"]["rows"]

    processed_path = resolve_path(
        Path(cfg["paths"]["processed_data_dir"])
        / cfg.get("preprocessing", {}).get("processed_dataset_filename", "processed_dataset.csv")
    )
    processed_rows = int(pd.read_csv(processed_path).shape[0]) if processed_path.is_file() else 0

    pipeline_path = resolve_path(
        Path(cfg["paths"]["processed_data_dir"])
        / cfg.get("preprocessing", {}).get("pipeline_filename", "preprocessing_pipeline.joblib")
    )
    encoded_features = 0
    if pipeline_path.is_file():
        pipeline = joblib.load(pipeline_path)
        encoded_features = len(pipeline.named_steps["column_transformer"].get_feature_names_out())

    imbalance_ratio = round(not_malaria_count / malaria_count, 2) if malaria_count else float("nan")

    rows = [
        {"metric": "Study setting", "value": "Hoima District, Uganda (rural health centres)"},
        {"metric": "Data source", "value": str(findings.get("source_file", "Kobo clinical export"))},
        {"metric": "Raw records", "value": raw_rows},
        {"metric": "Labelled diagnoses", "value": labelled},
        {"metric": "Malaria cases", "value": malaria_count},
        {"metric": "Not malaria cases", "value": not_malaria_count},
        {"metric": "Class imbalance ratio (negative:positive)", "value": imbalance_ratio},
        {"metric": "Clinical feature fields", "value": len(data_cfg["clinical_feature_columns"])},
        {"metric": "Records after preprocessing", "value": processed_rows},
        {"metric": "Encoded model features", "value": encoded_features},
        {"metric": "Target variable", "value": target_column},
        {"metric": "Temperature column present", "value": findings["numeric_checks"]["temperature_column_present"]},
    ]
    return pd.DataFrame(rows)


def build_table_4_2_missing_values(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    data_cfg = cfg["data"]
    df, _ = load_raw_dataframe(cfg)
    total = len(df)

    columns = list(data_cfg["clinical_feature_columns"]) + [data_cfg["target_column"]]
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in df.columns:
            continue
        missing_count = int(is_missing(df[column]).sum())
        rows.append(
            {
                "variable": column,
                "non_missing": total - missing_count,
                "missing": missing_count,
                "missing_percent": round(100 * missing_count / total, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_percent", ascending=False).reset_index(drop=True)


def build_table_4_3_feature_statistics(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    processed_path = resolve_path(
        Path(cfg["paths"]["processed_data_dir"])
        / cfg.get("preprocessing", {}).get("processed_dataset_filename", "processed_dataset.csv")
    )
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")

    df = pd.read_csv(processed_path)
    rows: list[dict[str, Any]] = []

    for column in ["Age", "Fever Duration (Days)"]:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "type": "numeric",
                "count": int(series.notna().sum()),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
                "mode_or_top_category": "—",
            }
        )

    categorical_columns = [c for c in model_feature_columns() if c not in {"Age", "Fever Duration (Days)", "visit_month"}]
    for column in categorical_columns:
        if column not in df.columns:
            continue
        series = df[column].fillna("<missing>")
        top = series.value_counts().head(1)
        top_category = str(top.index[0])
        top_count = int(top.iloc[0])
        rows.append(
            {
                "feature": column,
                "type": "categorical",
                "count": int(series.ne("<missing>").sum()),
                "mean": "—",
                "std": "—",
                "min": "—",
                "median": "—",
                "max": "—",
                "mode_or_top_category": f"{top_category} (n={top_count})",
            }
        )

    return pd.DataFrame(rows)


def build_table_4_4_baseline_results(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path = resolve_path(Path(cfg["paths"]["results_dir"]) / cfg.get("training", {}).get("comparison_filename", "baseline_comparison.csv"))
    if not path.is_file():
        raise FileNotFoundError(f"Baseline comparison not found at {path}. Run train-baselines first.")

    baseline = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, record in baseline.iterrows():
        rows.append(
            {
                "model": record["model_name"],
                "accuracy": _format_mean_std(record["accuracy_mean"], record["accuracy_std"]),
                "precision": _format_mean_std(record["precision_mean"], record["precision_std"]),
                "recall_sensitivity": _format_mean_std(record["recall_mean"], record["recall_std"]),
                "specificity": _format_mean_std(record["specificity_mean"], record["specificity_std"]),
                "roc_auc": _format_mean_std(record["roc_auc_mean"], record["roc_auc_std"]),
                "f1_score": _format_mean_std(record["f1_mean"], record["f1_std"]),
                "balanced_accuracy": _format_mean_std(record["balanced_accuracy_mean"], record["balanced_accuracy_std"]),
            }
        )
    return pd.DataFrame(rows)


def build_table_4_5_optimized_results(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path = resolve_path(
        Path(cfg["paths"]["results_dir"])
        / cfg.get("hyperparameter_optimization", {}).get("after_filename", "hyperparameter_after_optimization.csv")
    )
    if not path.is_file():
        raise FileNotFoundError(f"Optimized results not found at {path}. Run optimize-models first.")

    optimized = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, record in optimized.iterrows():
        rows.append(
            {
                "model": record["model_name"],
                "best_hyperparameters": record.get("best_params", ""),
                "accuracy": _format_ci(record["accuracy_mean"], record["accuracy_ci_low"], record["accuracy_ci_high"]),
                "precision": _format_ci(record["precision_mean"], record["precision_ci_low"], record["precision_ci_high"]),
                "recall_sensitivity": _format_ci(record["recall_mean"], record["recall_ci_low"], record["recall_ci_high"]),
                "specificity": _format_ci(record["specificity_mean"], record["specificity_ci_low"], record["specificity_ci_high"]),
                "roc_auc": _format_ci(record["roc_auc_mean"], record["roc_auc_ci_low"], record["roc_auc_ci_high"]),
                "f1_score": _format_ci(record["f1_mean"], record["f1_ci_low"], record["f1_ci_high"]),
                "balanced_accuracy": _format_ci(
                    record["balanced_accuracy_mean"],
                    record["balanced_accuracy_ci_low"],
                    record["balanced_accuracy_ci_high"],
                ),
                "search_time_seconds": round(float(record.get("search_time_seconds", 0)), 2),
            }
        )
    return pd.DataFrame(rows)


def build_table_4_6_model_comparison(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path = resolve_path(
        Path(cfg["paths"]["results_dir"])
        / cfg.get("hyperparameter_optimization", {}).get("comparison_filename", "hyperparameter_comparison.csv")
    )
    if not path.is_file():
        raise FileNotFoundError(f"Model comparison table not found at {path}. Run optimize-models first.")

    comparison = pd.read_csv(path)
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall (sensitivity)",
        "specificity": "Specificity",
        "roc_auc": "ROC AUC",
        "f1": "F1 score",
        "balanced_accuracy": "Balanced accuracy",
    }
    rows: list[dict[str, Any]] = []
    for _, record in comparison.iterrows():
        for metric, label in metric_labels.items():
            before = record.get(f"{metric}_before")
            after = record.get(f"{metric}_after")
            delta = record.get(f"{metric}_delta")
            rows.append(
                {
                    "model": record["model_name"],
                    "metric": label,
                    "before_optimization": round(float(before), 4) if pd.notna(before) else "—",
                    "after_optimization": round(float(after), 4) if pd.notna(after) else "—",
                    "delta": round(float(delta), 4) if pd.notna(delta) else "—",
                }
            )
    return pd.DataFrame(rows)


def build_table_4_7_edge_performance(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    baseline_path = resolve_path(
        Path(cfg["paths"]["results_dir"]) / cfg.get("training", {}).get("comparison_filename", "baseline_comparison.csv")
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline comparison not found at {baseline_path}.")

    baseline = pd.read_csv(baseline_path)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    best_model_name = resolve_best_model_name(cfg)

    single_patient_latency_ms: float | None = None
    try:
        X, _, _ = load_preprocessed_training_data(cfg)
        artifact = load_model_artifact(best_model_name, config=cfg)
        estimator = artifact["estimator"]
        sample = X[:1]
        start = time.perf_counter()
        estimator.predict(sample)
        if hasattr(estimator, "predict_proba"):
            estimator.predict_proba(sample)
        single_patient_latency_ms = (time.perf_counter() - start) * 1000
    except FileNotFoundError:
        pass

    rows: list[dict[str, Any]] = []
    for _, record in baseline.iterrows():
        model_name = record["model_name"]
        optimized_path = models_dir / "optimized" / f"{model_name}.joblib"
        baseline_model_path = models_dir / "baseline" / f"{model_name}.joblib"
        model_path = optimized_path if optimized_path.is_file() else baseline_model_path
        on_disk_mb = _serialize_model_size_mb(model_path) if model_path.is_file() else record["model_size_mb"]

        row = {
            "model": model_name,
            "model_size_mb": round(float(on_disk_mb), 4),
            "cv_prediction_time_ms": round(float(record["prediction_time_seconds_mean"]) * 1000, 3),
            "cv_prediction_time_std_ms": round(float(record["prediction_time_seconds_std"]) * 1000, 3),
            "peak_memory_mb": round(float(record["memory_usage_mb"]), 4),
            "training_time_seconds": round(float(record["training_time_seconds_mean"]), 3),
            "deployment_ready": optimized_path.is_file(),
        }
        if model_name == best_model_name and single_patient_latency_ms is not None:
            row["single_patient_latency_ms"] = round(single_patient_latency_ms, 3)
        else:
            row["single_patient_latency_ms"] = "—"
        rows.append(row)

    return pd.DataFrame(rows)


def _auto_figsize(num_rows: int, num_cols: int) -> tuple[float, float]:
    width = max(8.0, min(18.0, 1.2 * num_cols + 2))
    height = max(3.0, min(24.0, 0.38 * num_rows + 1.8))
    return width, height


def render_table_figure(
    table: pd.DataFrame,
    *,
    title: str,
    caption: str,
    output_path: Path,
) -> None:
    """Render a publication-quality PNG table with a thesis caption."""
    plt.rcParams.update(PUBLICATION_RC)
    display_df = table.copy()
    display_df.columns = [str(col).replace("_", " ").title() for col in display_df.columns]
    cell_text = display_df.astype(str).values.tolist()
    col_labels = list(display_df.columns)

    fig_width, fig_height = _auto_figsize(len(display_df), len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table_artist = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1.0, 1.35)

    for (row_idx, _), cell in table_artist.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2F4F6F")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F4F6F8")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", wrap=True, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_chapter4_tables(config: dict[str, Any] | None = None) -> Chapter4TablesResult:
    """Build all Chapter 4 tables in memory."""
    builders = {
        "table_4_1_dataset_summary": build_table_4_1_dataset_summary,
        "table_4_2_missing_values": build_table_4_2_missing_values,
        "table_4_3_feature_statistics": build_table_4_3_feature_statistics,
        "table_4_4_baseline_results": build_table_4_4_baseline_results,
        "table_4_5_optimized_results": build_table_4_5_optimized_results,
        "table_4_6_model_comparison": build_table_4_6_model_comparison,
        "table_4_7_edge_performance": build_table_4_7_edge_performance,
    }

    tables: dict[str, pd.DataFrame] = {}
    captions = {key: spec["caption"] for key, spec in TABLE_SPECS.items()}
    for key, builder in builders.items():
        tables[key] = builder(config)

    return Chapter4TablesResult(
        tables=tables,
        captions=captions,
        stats={"generated_at_utc": datetime.now(timezone.utc).isoformat(), "table_count": len(tables)},
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def write_chapter4_tables(
    output_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Chapter4TablesResult:
    """Generate CSV and PNG exports for all Chapter 4 tables."""
    cfg = config or load_config()
    chapter_cfg = _chapter4_config(cfg)
    base_dir = resolve_path(output_dir or chapter_cfg.get("output_dir", "reports/chapter4"))
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    base_dir.mkdir(parents=True, exist_ok=True)

    result = generate_chapter4_tables(config=cfg)
    manifest_tables: dict[str, dict[str, str]] = {}
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for key, table in result.tables.items():
        spec = TABLE_SPECS[key]
        csv_path = tables_dir / f"{key}.csv"
        png_path = figures_dir / f"{key}.png"
        table.to_csv(csv_path, index=False)
        render_table_figure(
            table,
            title=spec["title"],
            caption=spec["caption"],
            output_path=png_path,
        )
        manifest_tables[key] = {
            "title": spec["title"],
            "caption": spec["caption"],
            "csv": _display_path(csv_path),
            "figure": _display_path(png_path),
        }

    captions_path = base_dir / "captions.json"
    captions_path.write_text(json.dumps(result.captions, indent=2), encoding="utf-8")

    manifest = {
        "generated_at_utc": result.stats["generated_at_utc"],
        "output_dir": _display_path(base_dir),
        "tables": manifest_tables,
    }
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result.stats.update(
        {
            "output_dir": _display_path(base_dir),
            "tables_dir": _display_path(tables_dir),
            "figures_dir": _display_path(figures_dir),
            "manifest_path": _display_path(manifest_path),
            "captions_path": _display_path(captions_path),
        }
    )
    return result
