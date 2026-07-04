# Optimising Edge-Device Machine Learning Models for Malaria Diagnosis

Master's thesis project: a production-oriented machine learning system for **binary malaria diagnosis** from **tabular clinical data** collected at rural health centers in **Hoima District, Uganda**.

| Item | Value |
|------|-------|
| **Target** | `Final Confirmed Diagnosis (Malaria / Not Malaria)` |
| **Positive class** | `Malaria` |
| **Negative class** | `Not malaria` |
| **Modality** | Clinical intake / triage features (not blood-smear images) |
| **Raw records** | 291 patient visits (Kobo export, semicolon-separated CSV) |

## Project layout

```
.
├── config/                 # YAML configuration (paths, schema, column groups)
├── data/
│   ├── raw/                # Immutable source exports — never edit in place
│   └── processed/          # Cleaned, feature-ready datasets written by pipelines
├── notebooks/              # Exploratory analysis and thesis visualisations
├── src/                    # Installable Python package (all production logic)
│   ├── preprocessing/      # Loading, cleaning, validation
│   ├── features/           # Encoding and feature engineering
│   ├── models/             # Training and model selection
│   ├── evaluation/         # Metrics, CV, explainability, thesis tables
│   ├── deployment/         # Offline FastAPI prediction server
│   └── utils/              # Config loading and path resolution
├── models/                 # Serialized trained model artifacts (joblib)
├── reports/                # Generated thesis / experiment write-ups
├── figures/                # Plots saved from notebooks or evaluation scripts
├── results/                # Metrics tables, predictions, run manifests
├── tests/                  # Automated tests
├── main.py                 # CLI entry point
├── requirements.txt
└── pyproject.toml
```

## Step-by-step: run the full pipeline

All commands are run from the project root with the virtual environment activated.

### 1. Environment setup

```bash
cd "Edge Device Malaria Machine Learning Model"
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional boosting libraries (used if installed during baseline training):

```bash
pip install catboost xgboost lightgbm
```

### 2. Check project configuration

```bash
python main.py info
```

Prints project title, paths, and target column from `config/default.yaml`.

### 3. Validate raw data

```bash
python main.py validate
```

Checks that the Kobo CSV exists, required columns are present, and output directories exist.

### 4. Data quality audit (read-only)

```bash
python main.py audit
```

Writes:

- `reports/data_quality_report.md`
- `results/data_quality_summary.csv`

### 5. Preprocess data

```bash
python main.py preprocess
```

Fits the sklearn preprocessing pipeline and writes:

- `data/processed/processed_dataset.csv`
- `data/processed/preprocessing_pipeline.joblib`
- `reports/preprocessing_report.md`

### 6. Engineer clinical features (optional)

```bash
python main.py engineer-features
```

Creates triage-safe engineered features (separate from baseline training features):

- `data/processed/engineered_dataset.csv`
- `reports/feature_engineering_report.md`

> Baseline training uses the preprocessing pipeline output (56 encoded features), not the engineered dataset.

### 7. Clear past results (before retraining)

Run this **before** training again (for example after adding more data) so old models, metrics, and reports do not mix with the new run:

```bash
python main.py clean-artifacts
```

Deletes contents of:

- `models/` (baseline and optimized artifacts)
- `results/` (metrics CSVs and manifests)
- `figures/` (ROC curves, confusion matrices)
- `reports/` (training, explainability, and Chapter 4 outputs)

Preserves `data/raw/`, `data/processed/`, and `.gitkeep` placeholders.

Useful options:

```bash
python main.py clean-artifacts --dry-run              # preview only
python main.py clean-artifacts --verbose              # list each path
python main.py clean-artifacts --include-processed    # also clear data/processed/
```

Use `--include-processed` when you will re-run `preprocess` as well (recommended after a dataset update).

### 8. Train baseline models

```bash
python main.py train-baselines
```

Trains seven baseline classifiers with stratified 5-fold CV and writes:

- `models/baseline/*.joblib`
- `results/baseline_comparison.csv`
- `results/baseline_ranking.csv`
- `figures/` (ROC curves, confusion matrices)
- `reports/baseline_training_report.md`

### 9. Hyperparameter optimization

```bash
python main.py optimize-models
```

Runs `RandomizedSearchCV` on the top-ranked optimizable models and writes:

- `models/optimized/*.joblib`
- `results/hyperparameter_before_optimization.csv`
- `results/hyperparameter_after_optimization.csv`
- `results/hyperparameter_comparison.csv`
- `reports/hyperparameter_optimization_report.md`

### 10. Explain the best model

```bash
python main.py explain-model
```

Generates SHAP, permutation importance, PDPs, and symptom rankings:

- `reports/explainability/best_model_explainability.md`
- `reports/explainability/figures/`
- `reports/explainability/tables/`

### 11. Generate Chapter 4 thesis tables

```bash
python main.py generate-chapter4-tables
```

Exports all seven Chapter 4 tables as CSV and publication-quality PNG figures:

- `reports/chapter4/tables/` (CSV)
- `reports/chapter4/figures/` (PNG with captions)
- `reports/chapter4/manifest.json`

### 12. Export versioned TFLite model (React Native / Android)

Requires steps 5, 8, and 9 (preprocess + train + optimize) and TensorFlow:

```bash
pip install tensorflow
python main.py export-tflite --bump patch
```

Exports the **baseline rank-1** model when it is `logistic_regression` (override with `--model-name logistic_regression` if needed) to:

```text
models/tflite/
  manifest.json
  v1.0.0/
    model.tflite
    feature_spec.json
    metadata.json
```

Versioning options:

```bash
python main.py export-tflite --bump patch
python main.py export-tflite --bump minor
python main.py export-tflite --version 1.2.0 --release-notes "Retrain on expanded cohort"
```

Commit and push so the mobile app can update:

```bash
git add models/tflite/
git commit -m "release(tflite): v1.0.0 logistic_regression"
git push origin dev
```

#### React Native model update contract

The app should treat GitHub as the model update channel (offline-first):

1. **Bundle** a fallback `models/tflite/vX.Y.Z/` at build time.
2. **On launch / periodically**, fetch:
   `https://raw.githubusercontent.com/Edgar256/Optimised-Edge-Device-Machine-Learning-Model-For-Malaria-Diagnosis/dev/models/tflite/manifest.json`
3. If `latest_version` is newer than the local version and `min_app_version` is satisfied:
   - download `model.tflite` and `feature_spec.json` from the release paths in `manifest.releases`
   - verify `model_sha256` / `feature_spec_sha256`
   - store under the app documents directory and activate the new version
4. If the network is unavailable, keep using the local/bundled model.
5. Build the feature vector **only** from `feature_spec.json` (feature order, categories, imputers, defaults, risk thresholds).

Version policy: **patch** = retrain same schema; **minor** = compatible feature_spec changes; **major** = breaking feature layout (raise `min_app_version`).

### 13. Run the offline prediction API

Requires steps 5, 8, and 9 to have been completed (preprocessing pipeline + optimized model on disk).

```bash
python main.py serve-api
```

Starts the FastAPI server (default: `http://0.0.0.0:8000`).

**Model selection:** by default the API **auto-selects rank 1** from `results/baseline_ranking.csv` (composite clinical / edge score from `train-baselines`). It loads the **optimized** artifact from `models/optimized/` when present, otherwise the baseline. If ranking is missing, it falls back to highest `roc_auc_mean` in `results/hyperparameter_after_optimization.csv`. Check `GET /health` for the loaded `model_name`. To pin a specific model, set `api.model_name` in `config/default.yaml`.

| Endpoint | URL |
|----------|-----|
| Health check | `GET /health` |
| Predict | `POST /v1/predict` |
| Interactive docs | `GET /docs` |
| OpenAPI schema | `GET /openapi.json` |

Example prediction:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 12,
    "gender": "female",
    "temperature_celsius": 38.5,
    "symptoms": {
      "fever": true,
      "headache": true,
      "chills": true,
      "fever_duration_days": 3
    },
    "rdt": "not_performed",
    "microscopy": "not_performed"
  }'
```

For Android on the same LAN, use `http://<device-ip>:8000/v1/predict`. No cloud services are required.

Custom host/port:

```bash
python main.py serve-api --host 127.0.0.1 --port 8000
```

### 14. Run tests

```bash
pytest
```

## Full pipeline (one-liner sequence)

After setup, run the entire workflow in order:

```bash
python main.py validate
python main.py audit
python main.py preprocess
python main.py engineer-features
python main.py train-baselines
python main.py optimize-models
python main.py explain-model
python main.py generate-chapter4-tables
python main.py export-tflite --bump patch
python main.py serve-api
```

### Retrain after new data (same columns)

```bash
python main.py clean-artifacts --include-processed
python main.py validate
python main.py preprocess
python main.py train-baselines
python main.py optimize-models
python main.py explain-model
python main.py generate-chapter4-tables
python main.py export-tflite --bump patch
python main.py serve-api
```

Then commit `models/tflite/` and push so the React Native app can pick up the new version.

## CLI reference

| Command | Description |
|---------|-------------|
| `info` | Print project metadata and paths |
| `validate` | Validate raw data schema |
| `clean-artifacts` | Delete past models, results, figures, and reports |
| `audit` | Data quality audit report |
| `preprocess` | Fit preprocessing pipeline |
| `engineer-features` | Triage-safe feature engineering |
| `train-baselines` | Train and compare baseline models |
| `optimize-models` | RandomizedSearchCV on top models |
| `explain-model` | SHAP, permutation importance, PDPs |
| `generate-chapter4-tables` | Thesis tables (CSV + PNG) |
| `export-tflite` | Versioned TFLite export for React Native |
| `serve-api` | Offline FastAPI prediction server |

## Configuration

All paths and column definitions live in `config/default.yaml`. Clinical features are separated from:

- **Identifier columns** — patient codes and UUIDs (not model inputs)
- **Metadata columns** — Kobo form submission fields
- **Post-diagnosis columns** — RDT, microscopy, treatment (excluded to avoid label leakage at triage time)

## Design principles

1. **Reproducibility** — configuration-driven pipelines with a fixed random seed.
2. **Separation of concerns** — notebooks explore; `src/` implements.
3. **Edge readiness** — offline FastAPI server and resource profiling for rural devices.
4. **No in-place raw edits** — always write transformations to `data/processed/`.
