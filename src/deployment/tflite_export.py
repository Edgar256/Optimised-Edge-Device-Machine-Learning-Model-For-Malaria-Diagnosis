"""Export the selected optimized model to versioned TensorFlow Lite artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.explainability import (
    _optional_model_override,
    load_model_artifact,
    resolve_best_model_name,
)
from src.models.train import load_preprocessed_training_data
from src.preprocessing.common import categorical_feature_columns, numeric_feature_columns
from src.utils.config import load_config
from src.utils.paths import resolve_path

SUPPORTED_EXPORT_MODELS = {"logistic_regression"}
PARITY_TOLERANCE = 1e-4
MANIFEST_SCHEMA_VERSION = 1


@dataclass
class TfliteExportResult:
    """Artifacts written by a TFLite export run."""

    version: str
    model_name: str
    output_dir: Path
    model_path: Path
    feature_spec_path: Path
    metadata_path: Path
    manifest_path: Path
    max_parity_error: float
    stats: dict[str, Any] = field(default_factory=dict)


def _tflite_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("tflite", {})


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_semver(version: str) -> tuple[int, int, int]:
    parts = version.strip().lstrip("v").split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid semver version: {version!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _format_semver(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def bump_semver(version: str, bump: Literal["major", "minor", "patch"] = "patch") -> str:
    """Return the next semantic version."""
    major, minor, patch = _parse_semver(version)
    if bump == "major":
        return _format_semver(major + 1, 0, 0)
    if bump == "minor":
        return _format_semver(major, minor + 1, 0)
    if bump == "patch":
        return _format_semver(major, minor, patch + 1)
    raise ValueError(f"Unsupported bump type: {bump!r}")


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "latest_version": "0.0.0",
            "model_name": None,
            "updated_at_utc": None,
            "min_app_version": "1.0.0",
            "releases": {},
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _extract_lr_pipeline(estimator: Any) -> tuple[StandardScaler, LogisticRegression]:
    if isinstance(estimator, Pipeline):
        scaler = estimator.named_steps.get("scaler")
        model = estimator.named_steps.get("model")
        if isinstance(scaler, StandardScaler) and isinstance(model, LogisticRegression):
            return scaler, model
    if isinstance(estimator, LogisticRegression):
        raise ValueError(
            "Logistic regression export requires a Pipeline with StandardScaler + LogisticRegression."
        )
    raise ValueError(
        "TFLite export currently supports only logistic_regression "
        f"(Pipeline[StandardScaler, LogisticRegression]). Got: {type(estimator)!r}"
    )


def _bake_scaler_into_lr_weights(
    scaler: StandardScaler, model: LogisticRegression
) -> tuple[np.ndarray, float]:
    """Fold StandardScaler into LR weights: logit = x @ w + b with sigmoid(logit)."""
    mean = scaler.mean_.astype(np.float64)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float64)
    coef = model.coef_.astype(np.float64).ravel()
    intercept = float(model.intercept_.ravel()[0])
    # ((x - mean) / scale) @ coef + intercept
    # = x @ (coef / scale) + (intercept - mean @ (coef / scale))
    weights = (coef / scale).astype(np.float32)
    bias = float(intercept - np.dot(mean, coef / scale))
    return weights, bias


def convert_lr_pipeline_to_tflite(scaler: StandardScaler, model: LogisticRegression) -> bytes:
    """Convert a fitted scaler + logistic regression pipeline head to TFLite bytes."""
    import tensorflow as tf

    n_features = int(scaler.mean_.shape[0])
    weights, bias = _bake_scaler_into_lr_weights(scaler, model)
    # Pure TF constants avoid Keras variable / TFLite converter issues.
    kernel = tf.constant(weights.reshape(n_features, 1), dtype=tf.float32)
    intercept = tf.constant([[bias]], dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, n_features], dtype=tf.float32, name="features")])
    def predict_fn(features: tf.Tensor) -> tf.Tensor:
        logits = tf.matmul(features, kernel) + intercept
        return tf.nn.sigmoid(logits, name="malaria_probability")

    concrete_fn = predict_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], predict_fn)
    converter.optimizations = []
    return converter.convert()


def predict_tflite_proba(tflite_bytes: bytes, features: np.ndarray) -> np.ndarray:
    """Run TFLite inference; returns malaria probabilities shape (n_samples,)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    probabilities = []
    for row in features:
        interpreter.set_tensor(input_details["index"], row.reshape(1, -1))
        interpreter.invoke()
        probabilities.append(float(interpreter.get_tensor(output_details["index"])[0, 0]))
    return np.asarray(probabilities, dtype=np.float64)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def build_feature_spec(
    preprocessing_pipeline: Any,
    feature_names: list[str],
    *,
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Serialize preprocessing details needed by the React Native feature builder."""
    cfg = config
    api_cfg = cfg.get("api", {})
    column_transformer = preprocessing_pipeline.named_steps["column_transformer"]
    numeric_pipeline = column_transformer.named_transformers_["numeric"]
    categorical_pipeline = column_transformer.named_transformers_["categorical"]
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    encoder = categorical_pipeline.named_steps["encoder"]

    numeric_columns = numeric_feature_columns()
    categorical_columns = categorical_feature_columns()

    categories: dict[str, list[str]] = {}
    for column_name, category_values in zip(categorical_columns, encoder.categories_):
        categories[column_name] = [None if value is None else str(value) for value in category_values]

    return {
        "schema_version": 1,
        "model_name": model_name,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_imputer_statistics": _json_safe(numeric_imputer.statistics_),
        "categorical_imputer_statistics": _json_safe(categorical_imputer.statistics_),
        "one_hot_categories": categories,
        "handle_unknown": "ignore",
        "api_field_map": {
            "age": "Age",
            "gender": "Gender (Male/Female)",
            "symptoms.fever": "Fever (Yes/No)",
            "symptoms.headache": "Headache (Yes/No)",
            "symptoms.chills": "Chills (Yes/No)",
            "symptoms.vomiting": "Vomiting (Yes/No)",
            "symptoms.fatigue": "Fatigue (Yes/No)",
            "symptoms.anemia_signs": "Anemia Signs (Yes/No)",
            "symptoms.fever_duration_days": "Fever Duration (Days)",
            "symptoms.other": "Other Symptoms (Specify)",
            "health_facility_name": "Health Facility Name",
            "geographical_zone": "Geographical Zone",
            "season": "Season of Visit (Dry/Rainy)",
            "recent_travel": "Recent Travel (Yes/No)",
            "exposure_risk": "Exposure Risk (e.g., mosquito-prone area) (Yes/No)",
            "household_malaria_history": "Household Malaria History (Yes/No)",
        },
        "yes_no_encoding": {"true": "yes", "false": "no"},
        "defaults": api_cfg.get(
            "defaults",
            {
                "health_facility_name": "kasomoro_health_centre",
                "geographical_zone": "mbaraara",
                "season": "Rainy",
            },
        ),
        "risk_thresholds": api_cfg.get(
            "risk_thresholds",
            {
                "high_confidence": 0.75,
                "medium_confidence": 0.5,
                "fever_temperature_celsius": 38.0,
            },
        ),
        "decision_threshold": 0.5,
        "positive_label": cfg["data"]["target_positive_label"],
        "negative_label": cfg["data"]["target_negative_label"],
        "output": {
            "name": "malaria_probability",
            "dtype": "float32",
            "shape": [1, 1],
        },
        "notes": [
            "Build a float32 vector of length n_features in feature_names order.",
            "Unknown categorical levels must be all-zero one-hot columns (handle_unknown=ignore).",
            "Temperature, RDT, and microscopy are not model inputs; apply risk_category rules in the app.",
        ],
    }


def _load_metrics_for_model(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    results_dir = resolve_path(config["paths"]["results_dir"])
    after_path = results_dir / config.get("hyperparameter_optimization", {}).get(
        "after_filename", "hyperparameter_after_optimization.csv"
    )
    if after_path.is_file():
        import pandas as pd

        after = pd.read_csv(after_path)
        rows = after.loc[after["model_name"] == model_name]
        if not rows.empty:
            row = rows.iloc[0]
            return {
                "source": "hyperparameter_after_optimization",
                "roc_auc_mean": float(row["roc_auc_mean"]),
                "recall_mean": float(row["recall_mean"]),
                "f1_mean": float(row["f1_mean"]),
                "accuracy_mean": float(row["accuracy_mean"]),
            }

    ranking_path = results_dir / config.get("training", {}).get("ranking_filename", "baseline_ranking.csv")
    if ranking_path.is_file():
        import pandas as pd

        ranking = pd.read_csv(ranking_path)
        rows = ranking.loc[ranking["model_name"] == model_name]
        if not rows.empty:
            row = rows.iloc[0]
            return {
                "source": "baseline_ranking",
                "rank": int(row["rank"]),
                "rank_score": float(row["rank_score"]),
                "roc_auc_mean": float(row["roc_auc_mean"]),
                "recall_mean": float(row["recall_mean"]),
                "f1_mean": float(row["f1_mean"]),
            }
    return {}


def _resolve_export_model_name(config: dict[str, Any], model_name: str | None) -> str:
    tflite_cfg = _tflite_config(config)
    explicit = _optional_model_override(model_name) or _optional_model_override(tflite_cfg.get("model_name"))
    selected = explicit or resolve_best_model_name(config)
    if selected not in SUPPORTED_EXPORT_MODELS:
        raise ValueError(
            f"TFLite export supports only {sorted(SUPPORTED_EXPORT_MODELS)}; "
            f"selected model is {selected!r}. Pass --model-name logistic_regression "
            "or change baseline ranking / tflite.model_name."
        )
    return selected


def export_tflite_model(
    *,
    version: str | None = None,
    bump: Literal["major", "minor", "patch"] = "patch",
    model_name: str | None = None,
    config: dict[str, Any] | None = None,
    min_app_version: str = "1.0.0",
    release_notes: str = "",
) -> TfliteExportResult:
    """Export the selected optimized LR model to a versioned TFLite release."""
    cfg = config or load_config()
    tflite_cfg = _tflite_config(cfg)
    output_root = resolve_path(tflite_cfg.get("output_dir", "models/tflite"))
    manifest_path = output_root / "manifest.json"
    manifest = _read_manifest(manifest_path)

    selected_model = _resolve_export_model_name(cfg, model_name)
    next_version = version or bump_semver(str(manifest.get("latest_version") or "0.0.0"), bump=bump)
    _parse_semver(next_version)

    artifact = load_model_artifact(selected_model, config=cfg)
    estimator = artifact["estimator"]
    scaler, lr_model = _extract_lr_pipeline(estimator)

    pipeline_rel = artifact.get("preprocessing_pipeline_path", "data/processed/preprocessing_pipeline.joblib")
    pipeline_path = resolve_path(pipeline_rel)
    if not pipeline_path.is_file():
        pipeline_path = resolve_path(
            Path(cfg["paths"]["processed_data_dir"]) / "preprocessing_pipeline.joblib"
        )
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Preprocessing pipeline not found at {pipeline_path}")

    preprocessing_pipeline = joblib.load(pipeline_path)
    feature_names = list(
        artifact.get("feature_names")
        or preprocessing_pipeline.named_steps["column_transformer"].get_feature_names_out()
    )

    tflite_bytes = convert_lr_pipeline_to_tflite(scaler, lr_model)

    X, _, _ = load_preprocessed_training_data(cfg)
    sample = np.asarray(X[: min(32, len(X))], dtype=np.float32)
    sklearn_proba = estimator.predict_proba(sample)[:, 1]
    tflite_proba = predict_tflite_proba(tflite_bytes, sample)
    max_parity_error = float(np.max(np.abs(sklearn_proba - tflite_proba)))
    if max_parity_error > PARITY_TOLERANCE:
        raise RuntimeError(
            f"TFLite parity check failed: max abs error {max_parity_error:.6g} > {PARITY_TOLERANCE}"
        )

    feature_spec = build_feature_spec(
        preprocessing_pipeline,
        feature_names,
        config=cfg,
        model_name=selected_model,
    )

    release_dir = output_root / f"v{next_version}"
    release_dir.mkdir(parents=True, exist_ok=True)
    model_path = release_dir / "model.tflite"
    feature_spec_path = release_dir / "feature_spec.json"
    metadata_path = release_dir / "metadata.json"

    model_path.write_bytes(tflite_bytes)
    feature_spec_path.write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")

    model_sha = _sha256_file(model_path)
    feature_spec_sha = _sha256_file(feature_spec_path)
    created_at = datetime.now(timezone.utc).isoformat()
    metrics = _load_metrics_for_model(selected_model, cfg)

    metadata = {
        "version": next_version,
        "model_name": selected_model,
        "created_at_utc": created_at,
        "git_sha": _git_sha(),
        "n_features": len(feature_names),
        "max_parity_error": max_parity_error,
        "parity_tolerance": PARITY_TOLERANCE,
        "model_sha256": model_sha,
        "feature_spec_sha256": feature_spec_sha,
        "metrics": metrics,
        "release_notes": release_notes,
        "min_app_version": min_app_version,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    release_entry = {
        "model_path": _display_path(model_path),
        "feature_spec_path": _display_path(feature_spec_path),
        "metadata_path": _display_path(metadata_path),
        "model_sha256": model_sha,
        "feature_spec_sha256": feature_spec_sha,
        "min_app_version": min_app_version,
        "release_notes": release_notes,
        "created_at_utc": created_at,
    }
    releases = dict(manifest.get("releases") or {})
    releases[next_version] = release_entry
    manifest.update(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "latest_version": next_version,
            "model_name": selected_model,
            "updated_at_utc": created_at,
            "min_app_version": min_app_version,
            "releases": releases,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return TfliteExportResult(
        version=next_version,
        model_name=selected_model,
        output_dir=release_dir,
        model_path=model_path,
        feature_spec_path=feature_spec_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        max_parity_error=max_parity_error,
        stats={
            "output_dir": _display_path(release_dir),
            "manifest_path": _display_path(manifest_path),
            "model_path": _display_path(model_path),
            "feature_spec_path": _display_path(feature_spec_path),
            "metadata_path": _display_path(metadata_path),
            "model_sha256": model_sha,
            "feature_spec_sha256": feature_spec_sha,
            "n_features": len(feature_names),
        },
    )
