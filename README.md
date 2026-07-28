# Optimising Edge-Device Machine Learning Models for Malaria Diagnosis

Master's thesis project: a production-oriented machine learning system for **binary malaria diagnosis** from **tabular clinical data** collected at rural health centers in **Hoima District, Uganda**.

| Item | Value |
|------|-------|
| **Target** | `Final Confirmed Diagnosis (Malaria / Not Malaria)` |
| **Positive class** | `Malaria` |
| **Negative class** | `Not malaria` |
| **Modality** | Clinical intake / triage features (not blood-smear images) |
| **Raw records** | 903 patient visits (`PATIENT_RECORD_DATA_EXTRACTION_FORM.csv`, semicolon-separated Kobo export) |

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

#### Deploy to Render

One Render **Web Service** serves both the **FastAPI backend** and the **React admin dashboard** from the same URL. The build step compiles `frontend/`; FastAPI serves `frontend/dist/` with SPA fallback for `/login`, `/dashboard`, etc.

The API needs these versioned artifacts in the repo (committed after `optimize-models`):

- `models/optimized/logistic_regression.joblib`
- `data/processed/preprocessing_pipeline.joblib`

| Render setting | Value |
|----------------|-------|
| Python version | `3.12` (or use root `runtime.txt`) |
| Build command | `bash scripts/render-build.sh` |
| Start command | `bash scripts/render-start.sh` |

Or paste the equivalent commands directly in Render **Settings**:

```bash
# Build
pip install -r requirements.txt && cd frontend && npm ci && npm run build

# Start (must use $PORT — do not use python main.py serve-api)
uvicorn src.deployment.api:app --host 0.0.0.0 --port $PORT
```

Set **Python Version** to `3.12` (or add env var `PYTHON_VERSION=3.12.8`). Deploy the **`dev`** branch — it contains the model artifacts.

Or connect the repo to [`render.yaml`](render.yaml) at the project root — Render will pick up the same build/start commands automatically.

**Environment variables** (Render dashboard → Environment):

- `DATABASE_URL` — hosted MySQL connection string (**not** `localhost` on Render)
- `JWT_SECRET` — token signing secret
- `ADMIN_REGISTRATION_SECRET` — required for admin signup

Do **not** set `VITE_API_URL` for combined deploy — the dashboard calls `/v1/...` on the same origin.

After deploy:

- Admin UI: `https://<your-service>.onrender.com/login`
- API health: `GET https://<your-service>.onrender.com/health` → `"model_loaded": true`
- OpenAPI: `https://<your-service>.onrender.com/docs`

Tables are created/migrated automatically on startup, or run `python main.py init-db` once against your MySQL instance.

**MySQL on Render (CloudClusters / Cloudsters):** Use the hostname, port, and credentials from your CloudClusters dashboard — not a raw server IP or port 3306.

Provider docs example:

```bash
mysql -h mysql-202816-0.cloudclusters.net -P 19889 -u admin -p<Password>
```

Convert to `DATABASE_URL` for Render **Environment**:

```
mysql://admin:YOUR_PASSWORD@mysql-202816-0.cloudclusters.net:19889/malaria_data_db?ssl=true&ssl_verify=false
```

| Field | CloudClusters value |
|-------|---------------------|
| Host | `mysql-202816-0.cloudclusters.net` (from your dashboard) |
| Port | `19889` (not 3306) |
| User | `admin` |
| Database | `malaria_data_db` |
| SSL | `?ssl=true&ssl_verify=false` (self-signed cert) |

Do **not** use `163.123.183.83`, port `3306`, or user `m4l4r14_us3r`.

**Test before deploying:**

```bash
export DATABASE_URL='mysql://admin:YOUR_PASSWORD@mysql-202816-0.cloudclusters.net:19889/malaria_data_db?ssl=true&ssl_verify=false'
python main.py check-db
```

Paste the same URL into Render → Environment → `DATABASE_URL` → Save → Manual Deploy.

**1045 Access denied:** Wrong password — copy the exact `admin` password from CloudClusters.

**2003 Can't connect:** Wrong host/port, or add `?ssl=true&ssl_verify=false`.

### 14. User accounts and patient records

Requires `DATABASE_URL` in `.env` (MySQL) and a one-time table setup:

```bash
cp .env.example .env   # edit DATABASE_URL, API_URL, JWT_SECRET, ADMIN_REGISTRATION_SECRET
python main.py init-db
python main.py serve-api
```

Each user has a `user_type`: `MEDICAL_PERSONNEL` (default, mobile app) or `ADMIN` (admin console).

**Mobile app (medical personnel)** — use `API_URL` with:

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/auth/signup` | POST | No | Register medical personnel |
| `/v1/auth/login` | POST | No | Login; `"remember_me": true` for a 30-day token |
| `/v1/auth/me` | GET | Bearer | Current user profile |
| `/v1/patients` | POST | Bearer | Create patient visit (all Kobo CSV fields + latitude/longitude) |
| `/v1/patients` | GET | Bearer | List your patient records |
| `/v1/patients/{id}` | GET/PUT/DELETE | Bearer | Read, update, or delete a record |

**Admin console** — separate registration and login:

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/v1/auth/admin/register` | POST | No | Register admin (requires `admin_registration_secret` matching `.env`) |
| `/v1/auth/admin/login` | POST | No | Admin login only |
| `/v1/auth/admin/me` | GET | Bearer (admin) | Admin profile |

Medical personnel cannot use admin login, and admins cannot use mobile login.

Medical signup example:

```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Jane",
    "last_name": "Nurse",
    "email": "jane@example.com",
    "phone": "+256700000001",
    "job_title": "Clinical Officer",
    "health_facility_name": "Kasomoro health centre",
    "password": "securepass123"
  }'
```

Login with remember-me:

```bash
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "jane@example.com", "password": "securepass123", "remember_me": true}'
```

Use the returned `access_token` as `Authorization: Bearer <token>` for patient endpoints.

Admin registration example:

```bash
curl -X POST http://localhost:8000/v1/auth/admin/register \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Alex",
    "last_name": "Admin",
    "email": "admin@example.com",
    "phone": "+256700000099",
    "job_title": "System Administrator",
    "password": "securepass123",
    "admin_registration_secret": "your-ADMIN_REGISTRATION_SECRET-from-env"
  }'
```

Admin login:

```bash
curl -X POST http://localhost:8000/v1/auth/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "securepass123", "remember_me": true}'
```

**Existing databases:** if you created tables before `user_type` was added, run:

```sql
ALTER TABLE users ADD COLUMN user_type VARCHAR(32) NOT NULL DEFAULT 'MEDICAL_PERSONNEL';
```

Custom host/port:

```bash
python main.py serve-api --host 127.0.0.1 --port 8000
```

### 15. Admin React dashboard

The `frontend/` app is an admin console for registration, login, and district oversight.

**One command (API + dashboard):**

```bash
python main.py dev
```

Starts the FastAPI backend on `http://127.0.0.1:8000` and the React dashboard on `http://localhost:5173`. Press `Ctrl+C` to stop both. Add `--reload` to auto-reload API code changes.

Or run separately:

```bash
# Terminal 1 — API
python main.py serve-api

# Terminal 2 — dashboard (proxies /v1 to localhost:8000)
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open `http://localhost:5173` — register with your `ADMIN_REGISTRATION_SECRET`, then sign in.

| Page | Route | Description |
|------|-------|-------------|
| Login | `/login` | Admin sign-in with remember-me |
| Register | `/register` | Create admin account |
| Overview | `/dashboard` | Patient counts, malaria stats, model status |
| Patients | `/dashboard/patients` | Searchable list of all visits |
| Patient detail | `/dashboard/patients/:id` | Full Kobo field view + GPS |
| Users | `/dashboard/users` | Medical personnel and admins |
| Profile | `/dashboard/profile` | Signed-in admin account |

Production build (also run automatically on Render):

```bash
cd frontend && npm ci && npm run build
```

When `frontend/dist/` exists, `python main.py serve-api` serves the dashboard at `/` on the same port as the API — same behavior as Render production.

### 16. Run tests

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
| `dev` | Run API + React admin dashboard together |
| `init-db` | Create MySQL tables for users and patient records |

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
