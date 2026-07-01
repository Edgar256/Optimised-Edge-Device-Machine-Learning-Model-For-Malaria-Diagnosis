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
│   ├── evaluation/         # Metrics, CV, edge-device benchmarking
│   ├── deployment/         # Export and on-device inference packaging
│   └── utils/              # Config loading and path resolution
├── models/                 # Serialized trained model artifacts (joblib, ONNX, etc.)
├── reports/                # Generated thesis / experiment write-ups
├── figures/                # Plots saved from notebooks or evaluation scripts
├── results/                # Metrics tables, predictions, run manifests
├── tests/                  # Automated tests
├── main.py                 # CLI entry point
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py info
python main.py validate
pytest
```

## Configuration

All paths and column definitions live in `config/default.yaml`. Clinical features are separated from:

- **Identifier columns** — patient codes and UUIDs (not model inputs)
- **Metadata columns** — Kobo form submission fields
- **Post-diagnosis columns** — RDT, microscopy, treatment (excluded to avoid label leakage at triage time)

## Design principles

1. **Reproducibility** — configuration-driven pipelines with a fixed random seed.
2. **Separation of concerns** — notebooks explore; `src/` implements.
3. **Edge readiness** — `src/deployment/` will host lightweight export paths for rural devices.
4. **No in-place raw edits** — always write transformations to `data/processed/`.

## Status

Project scaffolding is in place. Model training, feature pipelines, and deployment export are not yet implemented.
