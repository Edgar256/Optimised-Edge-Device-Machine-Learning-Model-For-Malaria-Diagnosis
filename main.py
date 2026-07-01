#!/usr/bin/env python3
"""Command-line entry point for the malaria edge ML project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.features.engineering import write_feature_engineering_outputs
from src.preprocessing.audit import write_audit_outputs
from src.preprocessing.pipeline import write_preprocessing_outputs
from src.utils.config import get_raw_data_path, load_config
from src.utils.paths import find_project_root, resolve_path


def _cmd_info(_: argparse.Namespace) -> int:
    root = find_project_root()
    config = load_config()
    raw_path = get_raw_data_path(config)

    print(config["project"]["title"])
    print(f"Project root : {root}")
    print(f"Raw dataset  : {raw_path}")
    print(f"Target column: {config['data']['target_column']}")
    return 0


def _cmd_validate(_: argparse.Namespace) -> int:
    config = load_config()
    raw_path = get_raw_data_path(config)
    data_cfg = config["data"]

    if not raw_path.is_file():
        print(f"ERROR: Raw dataset not found at {raw_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(
        raw_path,
        sep=data_cfg["csv_separator"],
        encoding=data_cfg["encoding"],
    )

    required_columns = {
        data_cfg["target_column"],
        *data_cfg["clinical_feature_columns"],
        *data_cfg["identifier_columns"],
        *data_cfg["metadata_columns"],
        *data_cfg["post_diagnosis_columns"],
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        print("ERROR: Raw dataset is missing expected columns:", file=sys.stderr)
        for column in missing:
            print(f"  - {column}", file=sys.stderr)
        return 1

    target = data_cfg["target_column"]
    labelled = df[target].dropna()
    labelled = labelled[labelled.astype(str).str.strip() != ""]
    value_counts = labelled.value_counts(dropna=False)

    print(f"Rows loaded          : {len(df)}")
    print(f"Labelled diagnoses   : {len(labelled)}")
    print("Target distribution  :")
    for label, count in value_counts.items():
        print(f"  {label!r}: {count}")

    for directory_key in ("processed_data_dir", "models_dir", "results_dir", "figures_dir", "reports_dir"):
        directory = resolve_path(config["paths"][directory_key])
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory    : {directory}")

    print("Validation passed.")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    audit = write_audit_outputs(
        report_path=args.report,
        summary_path=args.summary,
    )
    print(f"Report written : {resolve_path(args.report)}")
    print(f"Summary written: {resolve_path(args.summary)}")
    print(f"Rows audited   : {audit.findings['shape']['rows']}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    result = write_preprocessing_outputs(report_path=args.report)
    print(f"Processed dataset : {result.stats['processed_dataset_path']}")
    print(f"Pipeline saved    : {result.stats['pipeline_path']}")
    print(f"Report written    : {result.stats['report_path']}")
    print(f"Rows in / rows out: {result.stats['initial_rows']} -> {result.stats['final_rows']}")
    print(f"Feature columns   : {len(result.feature_names)}")
    return 0


def _cmd_engineer_features(args: argparse.Namespace) -> int:
    result = write_feature_engineering_outputs(report_path=args.report)
    print(f"Engineered dataset : {result.stats['output_path']}")
    print(f"Report written     : {result.stats['report_path']}")
    print(f"Rows engineered    : {result.stats['output_rows']}")
    print(f"New features       : {result.stats['engineered_feature_count']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Malaria edge-device ML — project utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Print project metadata and paths.")
    info_parser.set_defaults(func=_cmd_info)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate raw data schema and ensure output directories exist.",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Run read-only data quality audit and write report artifacts.",
    )
    audit_parser.add_argument(
        "--report",
        default="reports/data_quality_report.md",
        help="Path for the markdown data quality report.",
    )
    audit_parser.add_argument(
        "--summary",
        default="results/data_quality_summary.csv",
        help="Path for the per-column summary statistics CSV.",
    )
    audit_parser.set_defaults(func=_cmd_audit)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Fit preprocessing pipeline and write processed dataset artifacts.",
    )
    preprocess_parser.add_argument(
        "--report",
        default="reports/preprocessing_report.md",
        help="Path for the preprocessing report.",
    )
    preprocess_parser.set_defaults(func=_cmd_preprocess)

    engineer_parser = subparsers.add_parser(
        "engineer-features",
        help="Create triage-safe clinical features from the processed dataset.",
    )
    engineer_parser.add_argument(
        "--report",
        default="reports/feature_engineering_report.md",
        help="Path for the feature engineering report.",
    )
    engineer_parser.set_defaults(func=_cmd_engineer_features)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
