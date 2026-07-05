#!/usr/bin/env python3
"""Command-line entry point for the malaria edge ML project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.deployment.api import create_app
from src.deployment.tflite_export import export_tflite_model
from src.evaluation.chapter4_tables import write_chapter4_tables
from src.evaluation.explainability import write_explainability_outputs
from src.features.engineering import write_feature_engineering_outputs
from src.models.hyperparameter_search import write_hyperparameter_optimization_outputs
from src.models.train import write_baseline_training_outputs
from src.preprocessing.audit import write_audit_outputs
from src.preprocessing.pipeline import write_preprocessing_outputs
from src.utils.cleanup import clean_training_artifacts
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


def _cmd_train_baselines(args: argparse.Namespace) -> int:
    result = write_baseline_training_outputs(report_path=args.report)
    print(f"Models trained : {result.stats['model_count']}")
    print(f"Best model     : {result.stats['best_model']}")
    print(f"Comparison     : {result.stats['comparison_path']}")
    print(f"Ranking        : {result.stats['ranking_path']}")
    print(f"Report         : {result.stats['report_path']}")
    return 0


def _cmd_optimize_models(args: argparse.Namespace) -> int:
    result = write_hyperparameter_optimization_outputs(report_path=args.report)
    print(f"Models optimized : {', '.join(result.stats['models_optimized'])}")
    print(f"Before table     : {result.stats['before_path']}")
    print(f"After table      : {result.stats['after_path']}")
    print(f"Comparison       : {result.stats['comparison_path']}")
    print(f"Report           : {result.stats['report_path']}")
    return 0


def _cmd_explain_model(args: argparse.Namespace) -> int:
    result = write_explainability_outputs(report_path=args.report)
    print(f"Model explained  : {result.model_name}")
    print(f"Report written   : {result.stats['report_path']}")
    print(f"Figures          : {result.stats['figures_dir']}")
    print(f"Tables           : {result.stats['tables_dir']}")
    if result.symptom_ranking.empty:
        print("Symptom ranking  : no symptom features found")
    else:
        top = result.symptom_ranking.iloc[0]
        print(f"Top symptom      : {top['symptom']}")
    return 0


def _cmd_generate_chapter4_tables(args: argparse.Namespace) -> int:
    result = write_chapter4_tables(output_dir=args.output_dir)
    print(f"Chapter 4 tables : {result.stats['table_count']}")
    print(f"CSV output       : {result.stats['tables_dir']}")
    print(f"PNG output       : {result.stats['figures_dir']}")
    print(f"Manifest         : {result.stats['manifest_path']}")
    return 0


def _cmd_clean_artifacts(args: argparse.Namespace) -> int:
    result = clean_training_artifacts(
        include_processed=args.include_processed,
        dry_run=args.dry_run,
    )
    action = "Would delete" if result.dry_run else "Deleted"
    print(f"{action} {len(result.deleted)} path(s)")
    if result.include_processed:
        print("Included        : data/processed/")
    else:
        print("Preserved       : data/raw/, data/processed/")
    if args.verbose:
        for path in result.deleted:
            print(f"  - {path}")
    if result.dry_run:
        print("Dry run only — no files were removed. Re-run without --dry-run to delete.")
    return 0


def _cmd_export_tflite(args: argparse.Namespace) -> int:
    config = load_config()
    tflite_cfg = config.get("tflite", {})
    try:
        result = export_tflite_model(
            version=args.version,
            bump=args.bump,
            model_name=args.model_name,
            config=config,
            min_app_version=args.min_app_version or tflite_cfg.get("min_app_version", "1.0.0"),
            release_notes=args.release_notes or "",
        )
    except (ModuleNotFoundError, ImportError) as exc:
        if "tensorflow" in str(exc).lower():
            print(
                "ERROR: TensorFlow is required for TFLite export. "
                "Install with: pip install 'malaria-edge-ml[tflite]' or pip install tensorflow",
                file=sys.stderr,
            )
            return 1
        raise
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"TFLite version   : v{result.version}")
    print(f"Model            : {result.model_name}")
    print(f"Release dir      : {result.stats['output_dir']}")
    print(f"Manifest         : {result.stats['manifest_path']}")
    print(f"Model SHA-256    : {result.stats['model_sha256']}")
    print(f"Parity max error : {result.max_parity_error:.2e}")
    print("Commit models/tflite/ and push so the React Native app can update.")
    return 0


def _cmd_serve_api(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    config = load_config()
    api_cfg = config.get("api", {})
    port = args.port or int(os.environ.get("PORT", api_cfg.get("port", 8000)))
    default_host = "0.0.0.0" if os.environ.get("PORT") else api_cfg.get("host", "0.0.0.0")
    host = args.host or default_host

    print(f"Starting offline API on http://{host}:{port}")
    print(f"Health check     : http://{host}:{port}/health")
    print(f"Predict endpoint : http://{host}:{port}/v1/predict")
    print(f"OpenAPI docs     : http://{host}:{port}/docs")
    uvicorn.run("src.deployment.api:app", host=host, port=port, reload=args.reload)
    return 0


def _cmd_init_db(_: argparse.Namespace) -> int:
    try:
        from src.database.session import init_db
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    applied = init_db()
    print("Database tables created successfully.")
    for message in applied:
        print(f"Applied migration: {message}")
    return 0


def _cmd_dev(args: argparse.Namespace) -> int:
    """Start the FastAPI backend and React admin dashboard together."""
    import os
    import shutil
    import signal
    import subprocess
    import time

    root = find_project_root()
    frontend_dir = root / "frontend"
    if not frontend_dir.is_dir():
        print(f"ERROR: Frontend directory not found at {frontend_dir}", file=sys.stderr)
        return 1
    if shutil.which("npm") is None:
        print("ERROR: npm is not installed or not on PATH.", file=sys.stderr)
        return 1

    config = load_config()
    api_cfg = config.get("api", {})
    host = args.host or api_cfg.get("host", "127.0.0.1")
    port = args.port or int(os.environ.get("PORT", api_cfg.get("port", 8000)))
    dashboard_port = args.dashboard_port

    if not (frontend_dir / "node_modules").is_dir():
        print("Installing frontend dependencies…")
        install = subprocess.run(["npm", "install"], cwd=frontend_dir)
        if install.returncode != 0:
            return install.returncode

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.deployment.api:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if args.reload:
        api_cmd.append("--reload")

    frontend_cmd = ["npm", "run", "dev", "--", "--port", str(dashboard_port), "--host"]

    print("Starting development stack (Ctrl+C to stop both)")
    print(f"  API       : http://{host}:{port}")
    print(f"  Dashboard : http://localhost:{dashboard_port}")
    print(f"  API docs  : http://{host}:{port}/docs")

    processes: list[subprocess.Popen] = []

    def shutdown(_signum: int | None = None, _frame: object | None = None) -> None:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        api_proc = subprocess.Popen(api_cmd, cwd=root)
        frontend_proc = subprocess.Popen(frontend_cmd + [host], cwd=frontend_dir)
        processes.extend([api_proc, frontend_proc])

        while True:
            for proc in processes:
                code = proc.poll()
                if code is not None:
                    shutdown()
                    return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
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

    clean_parser = subparsers.add_parser(
        "clean-artifacts",
        help="Delete past models, results, figures, and reports before retraining.",
    )
    clean_parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Also delete data/processed/ (pipeline and processed CSVs).",
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List paths that would be deleted without removing them.",
    )
    clean_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each deleted path.",
    )
    clean_parser.set_defaults(func=_cmd_clean_artifacts)

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

    train_parser = subparsers.add_parser(
        "train-baselines",
        help="Train and evaluate baseline models with stratified cross-validation.",
    )
    train_parser.add_argument(
        "--report",
        default="reports/baseline_training_report.md",
        help="Path for the baseline training report.",
    )
    train_parser.set_defaults(func=_cmd_train_baselines)

    optimize_parser = subparsers.add_parser(
        "optimize-models",
        help="RandomizedSearchCV tuning for top baseline models.",
    )
    optimize_parser.add_argument(
        "--report",
        default="reports/hyperparameter_optimization_report.md",
        help="Path for the hyperparameter optimization report.",
    )
    optimize_parser.set_defaults(func=_cmd_optimize_models)

    explain_parser = subparsers.add_parser(
        "explain-model",
        help="Explain the best model with SHAP, permutation importance, and PDPs.",
    )
    explain_parser.add_argument(
        "--report",
        default="reports/explainability/best_model_explainability.md",
        help="Path for the explainability report.",
    )
    explain_parser.set_defaults(func=_cmd_explain_model)

    chapter4_parser = subparsers.add_parser(
        "generate-chapter4-tables",
        help="Generate Chapter 4 thesis tables as CSV and publication-quality PNG figures.",
    )
    chapter4_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: reports/chapter4).",
    )
    chapter4_parser.set_defaults(func=_cmd_generate_chapter4_tables)

    tflite_parser = subparsers.add_parser(
        "export-tflite",
        help="Export the selected optimized model to versioned TFLite artifacts for React Native.",
    )
    tflite_parser.add_argument(
        "--version",
        default=None,
        help="Explicit semver (e.g. 1.0.1). Overrides --bump.",
    )
    tflite_parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Semver bump when --version is omitted (default: patch).",
    )
    tflite_parser.add_argument(
        "--model-name",
        default=None,
        help="Model to export (default: baseline rank-1; must be logistic_regression).",
    )
    tflite_parser.add_argument(
        "--min-app-version",
        default=None,
        help="Minimum React Native app version required for this model release.",
    )
    tflite_parser.add_argument(
        "--release-notes",
        default="",
        help="Optional release notes stored in manifest/metadata.",
    )
    tflite_parser.set_defaults(func=_cmd_export_tflite)

    serve_parser = subparsers.add_parser(
        "serve-api",
        help="Run the offline FastAPI prediction server for edge/Android clients.",
    )
    serve_parser.add_argument("--host", default=None, help="Bind address (default from config).")
    serve_parser.add_argument("--port", type=int, default=None, help="Bind port (default from config).")
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only).",
    )
    serve_parser.set_defaults(func=_cmd_serve_api)

    init_db_parser = subparsers.add_parser(
        "init-db",
        help="Create database tables for users and patient records (requires DATABASE_URL).",
    )
    init_db_parser.set_defaults(func=_cmd_init_db)

    dev_parser = subparsers.add_parser(
        "dev",
        help="Run the FastAPI backend and React admin dashboard together.",
    )
    dev_parser.add_argument("--host", default="127.0.0.1", help="Bind address for API and dashboard.")
    dev_parser.add_argument("--port", type=int, default=None, help="API port (default from config).")
    dev_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=5173,
        help="Vite dashboard port (default: 5173).",
    )
    dev_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable API auto-reload (development only).",
    )
    dev_parser.set_defaults(func=_cmd_dev)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
