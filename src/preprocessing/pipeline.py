"""Sklearn preprocessing pipeline construction and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.common import (
    categorical_feature_columns,
    load_raw_dataframe,
    model_feature_columns,
    numeric_feature_columns,
)
from src.preprocessing.transformers import (
    CategoryNormalizerTransformer,
    DateFeatureExtractor,
    FeatureSelector,
    ImpossibleValueTransformer,
    RowFilterTransformer,
)
from src.utils.config import load_config
from src.utils.paths import resolve_path


@dataclass
class PreprocessingResult:
    """Artifacts and statistics from a preprocessing run."""

    cleaned_dataset: pd.DataFrame
    transformed_features: np.ndarray
    feature_names: list[str]
    pipeline: Pipeline
    stats: dict[str, Any] = field(default_factory=dict)


def _preprocessing_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("preprocessing", {})


def build_preprocessing_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build the sklearn preprocessing pipeline with a ColumnTransformer."""
    prep_cfg = _preprocessing_config(config)
    scale_numeric = bool(prep_cfg.get("scale_numeric", False))

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=20,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_feature_columns()),
            ("categorical", categorical_pipeline, categorical_feature_columns()),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "row_filter",
                RowFilterTransformer(
                    target_column=config["data"]["target_column"],
                    clinical_columns=config["data"]["clinical_feature_columns"],
                    test_patient_ids=prep_cfg.get("test_patient_ids", ["Test123"]),
                ),
            ),
            ("category_normalizer", CategoryNormalizerTransformer()),
            ("impossible_values", ImpossibleValueTransformer()),
            ("date_features", DateFeatureExtractor()),
            ("feature_selector", FeatureSelector(model_feature_columns())),
            ("column_transformer", column_transformer),
        ]
    )


def _encode_target(series: pd.Series, config: dict[str, Any]) -> pd.Series:
    data_cfg = config["data"]
    mapping = {
        data_cfg["target_positive_label"]: 1,
        data_cfg["target_negative_label"]: 0,
    }
    return series.map(mapping)


def _extract_pipeline_stats(pipeline: Pipeline, initial_rows: int, final_rows: int) -> dict[str, Any]:
    row_filter: RowFilterTransformer = pipeline.named_steps["row_filter"]
    impossible_values: ImpossibleValueTransformer = pipeline.named_steps["impossible_values"]
    column_transformer: ColumnTransformer = pipeline.named_steps["column_transformer"]

    return {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "row_filter": getattr(row_filter, "filter_stats_", {}),
        "impossible_values": getattr(impossible_values, "correction_stats_", {}),
        "scale_numeric": "scaler" in pipeline.named_steps["column_transformer"].named_transformers_["numeric"].named_steps,
        "output_feature_count": len(column_transformer.get_feature_names_out()),
    }


def _build_cleaned_dataset(
    filtered_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    data_cfg = config["data"]
    target_column = data_cfg["target_column"]
    identifier_columns = [column for column in data_cfg["identifier_columns"] if column in filtered_df.columns]

    cleaned = filtered_df[identifier_columns + model_feature_columns() + [target_column]].copy()
    cleaned["target_binary"] = _encode_target(cleaned[target_column], config)
    return cleaned


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def run_preprocessing(
    config: dict[str, Any] | None = None,
    processed_dataset_path: str | Path | None = None,
    pipeline_path: str | Path | None = None,
) -> PreprocessingResult:
    """Fit the preprocessing pipeline and write processed artifacts."""
    cfg = config or load_config()
    prep_cfg = _preprocessing_config(cfg)
    paths_cfg = cfg["paths"]

    processed_dataset_path = resolve_path(
        processed_dataset_path
        or Path(paths_cfg["processed_data_dir"]) / prep_cfg.get("processed_dataset_filename", "processed_dataset.csv")
    )
    pipeline_path = resolve_path(
        pipeline_path or Path(paths_cfg["processed_data_dir"]) / prep_cfg.get("pipeline_filename", "preprocessing_pipeline.joblib")
    )

    raw_df, _ = load_raw_dataframe(cfg)
    initial_rows = len(raw_df)

    pipeline = build_preprocessing_pipeline(cfg)
    pipeline.fit(raw_df)

    filtered_df = pipeline.named_steps["row_filter"].transform(raw_df)
    filtered_df = pipeline.named_steps["category_normalizer"].transform(filtered_df)
    filtered_df = pipeline.named_steps["impossible_values"].transform(filtered_df)
    filtered_df = pipeline.named_steps["date_features"].transform(filtered_df)

    cleaned_dataset = _build_cleaned_dataset(filtered_df, cfg)
    feature_matrix = pipeline.transform(raw_df)
    feature_names = pipeline.named_steps["column_transformer"].get_feature_names_out().tolist()

    processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataset.to_csv(processed_dataset_path, index=False)
    joblib.dump(pipeline, pipeline_path)

    stats = _extract_pipeline_stats(pipeline, initial_rows=initial_rows, final_rows=len(cleaned_dataset))
    stats["processed_dataset_path"] = _display_path(processed_dataset_path)
    stats["pipeline_path"] = _display_path(pipeline_path)
    stats["target_distribution"] = cleaned_dataset[cfg["data"]["target_column"]].value_counts().to_dict()
    stats["target_binary_distribution"] = cleaned_dataset["target_binary"].value_counts().to_dict()
    stats["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    return PreprocessingResult(
        cleaned_dataset=cleaned_dataset,
        transformed_features=feature_matrix,
        feature_names=feature_names,
        pipeline=pipeline,
        stats=stats,
    )


def render_preprocessing_report(result: PreprocessingResult, config: dict[str, Any] | None = None) -> str:
    """Render a markdown report describing preprocessing decisions and outcomes."""
    cfg = config or load_config()
    prep_cfg = _preprocessing_config(cfg)
    stats = result.stats
    row_filter = stats.get("row_filter", {})
    impossible = stats.get("impossible_values", {})

    lines = [
        "# Preprocessing Report",
        "",
        "Automated preprocessing of the raw Kobo clinical export. **Raw data was not modified.**",
        "",
        "## 1. Overview",
        "",
        f"- **Generated (UTC):** {stats.get('generated_at_utc')}",
        f"- **Input rows:** {stats.get('initial_rows')}",
        f"- **Output rows:** {stats.get('final_rows')}",
        f"- **Processed dataset:** `{stats.get('processed_dataset_path')}`",
        f"- **Fitted pipeline:** `{stats.get('pipeline_path')}`",
        f"- **Encoded feature count:** {len(result.feature_names)}",
        "",
        "## 2. Pipeline stages",
        "",
        "1. `RowFilterTransformer` — remove duplicate, unlabelled, test, and empty clinical rows.",
        "2. `CategoryNormalizerTransformer` — canonicalize yes/no, gender, season, facility, and zone values.",
        "3. `ImpossibleValueTransformer` — set out-of-range age and fever-duration values to missing.",
        "4. `DateFeatureExtractor` — derive `visit_month` from `Date of Visit`.",
        "5. `ColumnTransformer` — impute, encode categoricals, and optionally scale numeric features.",
        "",
        "## 3. Row filtering",
        "",
        "| Step | Rows removed |",
        "|------|--------------|",
        f"| Duplicate rows | {row_filter.get('duplicate_rows_removed', 0)} |",
        f"| Duplicate `_uuid` | {row_filter.get('duplicate_uuid_removed', 0)} |",
        f"| Missing target label | {row_filter.get('missing_target_removed', 0)} |",
        f"| Test patient IDs ({', '.join(prep_cfg.get('test_patient_ids', []))}) | {row_filter.get('test_rows_removed', 0)} |",
        f"| Empty clinical submissions | {row_filter.get('empty_clinical_removed', 0)} |",
        "",
        "## 4. Impossible value corrections",
        "",
        f"- **Age out of range (0–120):** {impossible.get('age_out_of_range', 0)} values set to missing",
        f"- **Fever duration out of range (0–60 days):** {impossible.get('fever_duration_out_of_range', 0)} values set to missing",
        "",
        "## 5. Missing value handling",
        "",
        "- Numeric features (`Age`, `Fever Duration (Days)`): median imputation inside the sklearn pipeline.",
        "- Categorical features: most-frequent imputation followed by one-hot encoding.",
        "- High-cardinality text (`Other Symptoms (Specify)`): normalized to lowercase; one-hot capped at 20 categories.",
        "",
        "## 6. Categorical normalization",
        "",
        "- Yes/No fields normalized to lowercase `yes` / `no`.",
        "- Gender normalized to `male` / `female`.",
        "- Health facilities mapped to canonical names (`mbaraara`, `kibaire`, `kasomoro_health_centre`, etc.).",
        "- Geographical zones mapped using spelling-variant rules (e.g. `Mbaraar`, `kib aire` → canonical forms).",
        "",
        "## 7. Feature encoding and scaling",
        "",
        f"- **Numeric scaling enabled:** {stats.get('scale_numeric')}",
        "- Categorical encoding: `OneHotEncoder(handle_unknown='ignore', max_categories=20)`.",
        "- Post-diagnosis columns (RDT, microscopy, treatment) are excluded to avoid label leakage.",
        "",
        "## 8. Target distribution after preprocessing",
        "",
        "| Label | Count |",
        "|-------|-------|",
    ]

    for label, count in stats.get("target_distribution", {}).items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "### Binary target (`target_binary`)",
            "",
            "| Value | Count |",
            "|-------|-------|",
        ]
    )
    for label, count in stats.get("target_binary_distribution", {}).items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## 9. Output feature names",
            "",
            ", ".join(f"`{name}`" for name in result.feature_names),
            "",
        ]
    )
    return "\n".join(lines)


def write_preprocessing_outputs(
    report_path: str | Path = "reports/preprocessing_report.md",
    config: dict[str, Any] | None = None,
) -> PreprocessingResult:
    """Run preprocessing and write the processed dataset, pipeline, and report."""
    result = run_preprocessing(config=config)
    report_file = resolve_path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_preprocessing_report(result, config=config), encoding="utf-8")
    result.stats["report_path"] = _display_path(report_file)
    return result
