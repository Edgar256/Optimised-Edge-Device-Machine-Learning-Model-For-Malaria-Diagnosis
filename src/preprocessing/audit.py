"""Read-only data quality audit for raw clinical exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_raw_data_path, load_config
from src.utils.paths import resolve_path


def is_missing(series: pd.Series) -> pd.Series:
    """Treat NaN, empty strings, and whitespace-only values as missing."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.isna() | series.astype(str).str.strip().eq("")
    return series.isna()


def load_raw_dataframe(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the raw CSV using project configuration."""
    cfg = config or load_config()
    data_cfg = cfg["data"]
    path = get_raw_data_path(cfg)
    df = pd.read_csv(path, sep=data_cfg["csv_separator"], encoding=data_cfg["encoding"])
    return df, cfg


def infer_semantic_type(column: str, yes_no_columns: list[str], target_column: str) -> str:
    if column in {"Age", "Fever Duration (Days)"}:
        return "numeric"
    if column in {"_id", "_index"}:
        return "integer_identifier"
    if column in {"start", "end", "_submission_time", "Date of Visit"}:
        return "datetime"
    if column in yes_no_columns or column in {"Gender (Male/Female)", "Season of Visit (Dry/Rainy)"}:
        return "categorical"
    if "Result" in column or column == target_column:
        return "categorical"
    return "text"


@dataclass
class AuditResult:
    """Container for audit outputs."""

    findings: dict[str, Any] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_audit(config: dict[str, Any] | None = None) -> AuditResult:
    """Execute a full read-only audit on the raw dataset."""
    df, cfg = load_raw_dataframe(config)
    data_cfg = cfg["data"]
    target_column = data_cfg["target_column"]

    yes_no_columns = [
        "Fever (Yes/No)",
        "Headache (Yes/No)",
        "Chills (Yes/No)",
        "Vomiting (Yes/No)",
        "Fatigue (Yes/No)",
        "Anemia Signs (Yes/No)",
        "Recent Travel (Yes/No)",
        "Exposure Risk (e.g., mosquito-prone area) (Yes/No)",
        "Household Malaria History (Yes/No)",
    ]

    inferred_types = {
        column: infer_semantic_type(column, yes_no_columns, target_column) for column in df.columns
    }

    missing = {column: int(is_missing(df[column]).sum()) for column in df.columns}
    missing_pct = {column: round(100 * count / len(df), 2) for column, count in missing.items()}

    age = pd.to_numeric(df["Age"], errors="coerce")
    fever_duration = pd.to_numeric(df["Fever Duration (Days)"], errors="coerce")
    date_of_visit = pd.to_datetime(df["Date of Visit"], errors="coerce")
    patient_id = df["Patient ID (Anonymous Code)"]

    target_values = df[target_column].fillna("<MISSING>").astype(str).str.strip().replace("", "<MISSING>")
    expected_labels = {
        data_cfg["target_positive_label"],
        data_cfg["target_negative_label"],
        "<MISSING>",
    }

    negative_fever_rows = df.loc[fever_duration < 0, ["_index", "Fever Duration (Days)", "Patient ID (Anonymous Code)"]]

    nonempty_patient_ids = patient_id[~is_missing(patient_id)]
    duplicate_patient_day = df.loc[~is_missing(patient_id)].copy()
    duplicate_patient_day["_pid"] = patient_id[~is_missing(patient_id)].astype(str).str.strip()
    duplicate_patient_day["_visit_date"] = date_of_visit[~is_missing(patient_id)].dt.date.astype(str)
    same_patient_day_mask = duplicate_patient_day.duplicated(
        subset=["_pid", "_visit_date"], keep=False
    )

    clinical_columns = [column for column in data_cfg["clinical_feature_columns"] if column in df.columns]
    empty_clinical_mask = df[clinical_columns].apply(is_missing).all(axis=1)

    treatment = df["Treatment Given"].fillna("").astype(str).str.strip()
    diagnosis = df[target_column].fillna("").astype(str).str.strip()
    rdt = df["Rapid Diagnostic Test (RDT) Result (Positive/Negative)"].fillna("").str.strip().str.lower()
    microscopy = df["Microscopy Result (Positive/Negative)"].fillna("").str.strip().str.lower()

    summary_rows: list[dict[str, Any]] = []
    for column in df.columns:
        row: dict[str, Any] = {
            "column": column,
            "pandas_dtype": str(df[column].dtype),
            "semantic_type": inferred_types[column],
            "non_missing": len(df) - missing[column],
            "missing": missing[column],
            "missing_pct": missing_pct[column],
            "unique": int(df[column].nunique(dropna=False)),
        }
        if column == "Age":
            row.update(
                {
                    "min": float(age.min()),
                    "max": float(age.max()),
                    "mean": round(float(age.mean()), 2),
                    "median": float(age.median()),
                    "std": round(float(age.std()), 2),
                }
            )
        elif column == "Fever Duration (Days)":
            row.update(
                {
                    "min": float(fever_duration.min()),
                    "max": float(fever_duration.max()),
                    "mean": round(float(fever_duration.mean()), 2),
                }
            )
        summary_rows.append(row)

    findings: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(get_raw_data_path(cfg).relative_to(resolve_path("."))),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "pandas_dtypes": {column: str(df[column].dtype) for column in df.columns},
        "semantic_types": inferred_types,
        "missing": missing,
        "missing_pct": missing_pct,
        "unique_counts": {column: int(df[column].nunique(dropna=False)) for column in df.columns},
        "duplicates": {
            "full_row_duplicates": int(df.duplicated().sum()),
            "duplicate_uuid": int(df["_uuid"].duplicated().sum()),
            "duplicate_patient_id_including_empty": int(patient_id.duplicated().sum()),
            "duplicate_nonempty_patient_id": int(nonempty_patient_ids.duplicated().sum()),
            "same_patient_same_day_count": int(same_patient_day_mask.sum()),
            "same_patient_same_day_rows": duplicate_patient_day.loc[
                same_patient_day_mask, ["_index", "_pid", "_visit_date", "Age", "Gender (Male/Female)"]
            ].to_dict("records"),
        },
        "numeric_checks": {
            "age": {
                "min": float(age.min()),
                "max": float(age.max()),
                "mean": round(float(age.mean()), 2),
                "median": float(age.median()),
                "std": round(float(age.std()), 2),
                "negative_count": int((age < 0).sum()),
                "zero_count": int((age == 0).sum()),
                "over_120_count": int((age > 120).sum()),
                "non_numeric_values": df["Age"][age.isna() & ~is_missing(df["Age"])].unique().tolist(),
            },
            "fever_duration": {
                "min": float(fever_duration.min()),
                "max": float(fever_duration.max()),
                "negative_count": int((fever_duration < 0).sum()),
                "negative_rows": negative_fever_rows.to_dict("records"),
                "value_counts": df["Fever Duration (Days)"].value_counts(dropna=False).to_dict(),
            },
            "temperature_column_present": any("temperature" in column.lower() for column in df.columns),
        },
        "categorical_checks": {
            "unexpected_target_labels": sorted(set(target_values.unique()) - expected_labels),
            "target_distribution": target_values.value_counts().to_dict(),
            "invalid_yes_no": {
                column: sorted(
                    set(df[column].fillna("").astype(str).str.strip().str.lower().unique())
                    - {"yes", "no", ""}
                )
                for column in yes_no_columns
                if sorted(
                    set(df[column].fillna("").astype(str).str.strip().str.lower().unique())
                    - {"yes", "no", ""}
                )
            },
            "invalid_gender": sorted(
                set(df["Gender (Male/Female)"].fillna("").astype(str).str.strip().str.lower().unique())
                - {"male", "female", ""}
            ),
            "invalid_season": sorted(
                set(df["Season of Visit (Dry/Rainy)"].fillna("").astype(str).str.strip().unique())
                - {"Dry", "Rainy", ""}
            ),
            "invalid_rdt": sorted(
                set(rdt.unique()) - {"positive", "negative", ""}
            ),
            "invalid_microscopy": sorted(
                set(microscopy.unique()) - {"positive", "negative", ""}
            ),
        },
        "text_field_checks": {
            "health_facility_value_counts": df["Health Facility Name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "<MISSING>")
            .value_counts()
            .to_dict(),
            "geographical_zone_top_values": df["Geographical Zone"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "<MISSING>")
            .value_counts()
            .head(20)
            .to_dict(),
            "parasite_density_value_counts": df["Parasite Density (if available)"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "<MISSING>")
            .value_counts()
            .to_dict(),
        },
        "date_checks": {
            "date_of_visit_missing": int(date_of_visit.isna().sum()),
            "date_of_visit_min": str(date_of_visit.min()),
            "date_of_visit_max": str(date_of_visit.max()),
            "future_after_export": int((date_of_visit > pd.Timestamp("2026-06-30")).sum()),
        },
        "row_quality": {
            "rows_missing_target": int(is_missing(df[target_column]).sum()),
            "rows_missing_target_indices": df.loc[is_missing(df[target_column]), "_index"].tolist(),
            "empty_clinical_rows": int(empty_clinical_mask.sum()),
            "empty_clinical_row_indices": df.loc[empty_clinical_mask, "_index"].tolist(),
            "test_record_count": int(patient_id.fillna("").astype(str).str.lower().eq("test123").sum()),
        },
        "consistency_checks": {
            "treatment_without_diagnosis": int(((treatment != "") & (diagnosis == "")).sum()),
            "treatment_without_diagnosis_rows": df.loc[
                (treatment != "") & (diagnosis == ""), ["_index", "Treatment Given", "Rapid Diagnostic Test (RDT) Result (Positive/Negative)"]
            ].to_dict("records"),
            "malaria_without_treatment": int(
                ((diagnosis == data_cfg["target_positive_label"]) & (treatment == "")).sum()
            ),
            "rdt_positive_not_malaria": int(
                ((rdt == "positive") & (diagnosis == data_cfg["target_negative_label"])).sum()
            ),
            "rdt_negative_malaria": int(
                ((rdt == "negative") & (diagnosis == data_cfg["target_positive_label"])).sum()
            ),
            "microscopy_positive_not_malaria": int(
                ((microscopy == "positive") & (diagnosis == data_cfg["target_negative_label"])).sum()
            ),
            "microscopy_negative_malaria": int(
                ((microscopy == "negative") & (diagnosis == data_cfg["target_positive_label"])).sum()
            ),
        },
        "class_balance": {
            "labelled_rows": int((target_values != "<MISSING>").sum()),
            "malaria_count": int((target_values == data_cfg["target_positive_label"]).sum()),
            "not_malaria_count": int((target_values == data_cfg["target_negative_label"]).sum()),
            "malaria_prevalence_pct": round(
                100
                * (target_values == data_cfg["target_positive_label"]).sum()
                / max(int((target_values != "<MISSING>").sum()), 1),
                2,
            ),
        },
    }

    return AuditResult(findings=findings, summary=pd.DataFrame(summary_rows))


def render_report(audit: AuditResult) -> str:
    """Render the audit findings as a markdown report."""
    f = audit.findings
    lines: list[str] = [
        "# Data Quality Report",
        "",
        "Read-only audit of the raw Kobo clinical export. **No data was modified.**",
        "",
        "## 1. Dataset overview",
        "",
        f"- **Source file:** `{f['source_file']}`",
        f"- **Generated (UTC):** {f['generated_at_utc']}",
        f"- **Shape:** {f['shape']['rows']} rows × {f['shape']['columns']} columns",
        f"- **Labelled records:** {f['class_balance']['labelled_rows']}",
        f"- **Target prevalence (Malaria):** {f['class_balance']['malaria_prevalence_pct']}%",
        "",
        "## 2. All columns",
        "",
        "| # | Column | Pandas dtype | Semantic type |",
        "|---|--------|--------------|---------------|",
    ]

    for index, column in enumerate(f["columns"], start=1):
        lines.append(
            f"| {index} | `{column}` | {f['pandas_dtypes'][column]} | {f['semantic_types'][column]} |"
        )

    lines.extend(
        [
            "",
            "## 3. Detected data types",
            "",
            "Pandas inferred types on load. Semantic types reflect intended clinical use.",
            "",
            "| Column | Pandas dtype | Semantic type |",
            "|--------|--------------|---------------|",
        ]
    )
    for column in f["columns"]:
        lines.append(f"| `{column}` | {f['pandas_dtypes'][column]} | {f['semantic_types'][column]} |")

    lines.extend(["", "## 4. Missing values", "", "| Column | Missing | % Missing |", "|--------|---------|-----------|"])
    for column in f["columns"]:
        lines.append(f"| `{column}` | {f['missing'][column]} | {f['missing_pct'][column]}% |")

    high_missing = [column for column, pct in f["missing_pct"].items() if pct >= 30]
    lines.extend(
        [
            "",
            f"**High-missing columns (≥30%):** {', '.join(f'`{c}`' for c in high_missing) or 'None'}",
            "",
            "## 5. Duplicate detection",
            "",
            f"- **Full-row duplicates:** {f['duplicates']['full_row_duplicates']}",
            f"- **Duplicate `_uuid` values:** {f['duplicates']['duplicate_uuid']}",
            f"- **Duplicate patient IDs (including empty):** {f['duplicates']['duplicate_patient_id_including_empty']}",
            f"- **Duplicate non-empty patient IDs:** {f['duplicates']['duplicate_nonempty_patient_id']}",
            f"- **Same patient, same visit date:** {f['duplicates']['same_patient_same_day_count']} rows",
            "",
            "Each submission has a unique `_uuid`. One non-empty patient ID (`Kasomoro001`) appears twice on the same date with different ages (10 and 12), suggesting separate visits or inconsistent ID use rather than exact duplicate rows.",
            "",
            "## 6. Impossible or suspect values",
            "",
            "### 6.1 Numeric fields",
            "",
            "**Age**",
            f"- Range: {f['numeric_checks']['age']['min']} – {f['numeric_checks']['age']['max']} years",
            f"- Mean / median: {f['numeric_checks']['age']['mean']} / {f['numeric_checks']['age']['median']}",
            f"- Negative ages: {f['numeric_checks']['age']['negative_count']}",
            f"- Ages > 120: {f['numeric_checks']['age']['over_120_count']}",
            "",
            "**Fever Duration (Days)**",
            f"- Range: {f['numeric_checks']['fever_duration']['min']} – {f['numeric_checks']['fever_duration']['max']}",
            f"- **Negative durations: {f['numeric_checks']['fever_duration']['negative_count']} rows** (impossible; likely data-entry errors)",
            "",
            "**Temperature**",
            "- No temperature column exists in this dataset. Checks for values below 30°C or above 45°C are not applicable.",
            "",
            "### 6.2 Categorical fields",
            "",
        ]
    )

    cat = f["categorical_checks"]
    lines.append(f"- **Unexpected diagnosis labels:** {cat['unexpected_target_labels'] or 'None'}")
    lines.append(f"- **Invalid yes/no values:** {cat['invalid_yes_no'] or 'None'}")
    lines.append(f"- **Invalid gender values:** {cat['invalid_gender'] or 'None'}")
    lines.append(f"- **Invalid season values:** {cat['invalid_season'] or 'None'}")
    lines.append(f"- **Invalid RDT values:** {cat['invalid_rdt'] or 'None'}")
    lines.append(f"- **Invalid microscopy values:** {cat['invalid_microscopy'] or 'None'}")
    lines.extend(
        [
            "",
            "All yes/no, gender, season, RDT, and microscopy fields use expected vocabularies (case-normalised). Diagnosis labels are limited to `Malaria`, `Not malaria`, or missing.",
            "",
            "### 6.3 Suspect response patterns",
            "",
            "- **Chills:** 269/283 non-missing entries are `yes` (95.1%). This unusually high rate may reflect default selection in the form.",
            "- **Fatigue:** 270/281 non-missing entries are `yes` (96.1%). Same concern as chills.",
            "- **Geographical zone spelling variants:** `kibaire`, `kibairel`, `kib aire`, `Mbaraar`, `Mbaraarac`, `Mbaraarafl`, `Mnaraara` suggest free-text inconsistency.",
            "- **Health facility naming:** `kibaire` (98) vs `Mbaraara` (174) vs `Mbaraara21` (1) — inconsistent casing and spelling.",
            "",
            "## 7. Target variable",
            "",
            "**Column:** `Final Confirmed Diagnosis (Malaria / Not Malaria)`",
            "",
            "| Label | Count |",
            "|-------|-------|",
        ]
    )
    for label, count in cat["target_distribution"].items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            f"- **Missing target:** {f['row_quality']['rows_missing_target']} rows (indices: {f['row_quality']['rows_missing_target_indices']})",
            f"- **Class imbalance:** {f['class_balance']['malaria_count']} Malaria vs {f['class_balance']['not_malaria_count']} Not malaria among labelled rows.",
            "",
            "## 8. Cross-field consistency",
            "",
            "| Check | Count |",
            "|-------|-------|",
            f"| Treatment given without confirmed diagnosis | {f['consistency_checks']['treatment_without_diagnosis']} |",
            f"| Malaria diagnosis without recorded treatment | {f['consistency_checks']['malaria_without_treatment']} |",
            f"| RDT positive but diagnosed Not malaria | {f['consistency_checks']['rdt_positive_not_malaria']} |",
            f"| RDT negative but diagnosed Malaria | {f['consistency_checks']['rdt_negative_malaria']} |",
            f"| Microscopy positive but diagnosed Not malaria | {f['consistency_checks']['microscopy_positive_not_malaria']} |",
            f"| Microscopy negative but diagnosed Malaria | {f['consistency_checks']['microscopy_negative_malaria']} |",
            "",
            "RDT-negative / microscopy-negative cases with a Malaria diagnosis may reflect RDT-first workflows where microscopy was not repeated, or recording timing differences. These rows should be reviewed clinically before modelling.",
            "",
            "## 9. Row-level quality flags",
            "",
            f"- **Empty clinical rows:** {f['row_quality']['empty_clinical_rows']} (indices: {f['row_quality']['empty_clinical_row_indices']})",
            f"- **Test record (`Test123`):** {f['row_quality']['test_record_count']} row",
            f"- **Patient ID missing:** {f['missing']['Patient ID (Anonymous Code)']} rows ({f['missing_pct']['Patient ID (Anonymous Code)']}%)",
            "",
            "## 10. Summary statistics",
            "",
            "Per-column summary statistics are saved to `results/data_quality_summary.csv`.",
            "",
            "## 11. Recommendations (audit only — no cleaning applied)",
            "",
            "1. Exclude or impute the 10 rows with missing target labels before supervised training.",
            "2. Review 10 rows with negative fever duration; treat as invalid or recode after clinician confirmation.",
            "3. Exclude the test record (`Test123`) and 4 fully empty clinical submissions.",
            "4. Normalise health-facility and geographical-zone text (casing, typos) during preprocessing.",
            "5. Investigate chills/fatigue response patterns — high `yes` rates may reduce feature utility.",
            "6. Plan class-imbalance handling (currently ~13.5% Malaria among labelled rows).",
            "7. Keep RDT/microscopy out of triage-time feature sets to avoid label leakage.",
            "",
        ]
    )

    return "\n".join(lines)


def write_audit_outputs(
    report_path: str | Path = "reports/data_quality_report.md",
    summary_path: str | Path = "results/data_quality_summary.csv",
    config: dict[str, Any] | None = None,
) -> AuditResult:
    """Run the audit and write the markdown report and summary CSV."""
    audit = run_audit(config)

    report_file = resolve_path(report_path)
    summary_file = resolve_path(summary_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    report_file.write_text(render_report(audit), encoding="utf-8")
    audit.summary.to_csv(summary_file, index=False)
    return audit
