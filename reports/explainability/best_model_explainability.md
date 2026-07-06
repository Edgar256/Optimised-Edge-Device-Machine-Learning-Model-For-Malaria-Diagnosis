# Deployed Model Explainability Report

- **Model:** `logistic_regression` (deployed edge model)
- **Generated (UTC):** 2026-07-06T08:04:43.905941+00:00
- **Samples analysed:** 844
- **Features:** 71

## Model performance context

The deployed model achieved ROC AUC **0.983** (95% CI [0.966, 0.999]), recall **0.883**, and F1 **0.920** after hyperparameter optimization.

## Symptom influence summary

Based on **permutation importance**, the symptoms most associated with malaria diagnosis are:

- **Fever (Yes/No)** (aggregate score 0.0048)
- **Headache (Yes/No)** (aggregate score 0.0000)
- **Chills (Yes/No)** (aggregate score 0.0000)

Positive associations for binary symptom features indicate that reporting the symptom increases predicted malaria risk relative to the reference encoded category.

Based on **SHAP values**, the symptoms most associated with malaria diagnosis are:

- **Fever (Yes/No)** (aggregate score 0.1616)
- **Headache (Yes/No)** (aggregate score 0.0000)
- **Chills (Yes/No)** (aggregate score 0.0000)

Positive associations for binary symptom features indicate that reporting the symptom increases predicted malaria risk relative to the reference encoded category.

## Top overall features (permutation importance)

| rank | feature | importance_mean | importance_std |
| --- | --- | --- | --- |
| 1 | Fever Duration (Days) | 0.1669616693799023 | 0.00617314708516018 |
| 2 | Household Malaria History (Yes/No)_yes | 0.01467367174781702 | 0.0041682458515161355 |
| 3 | Fever (Yes/No)_no | 0.004801687139262989 | 0.0036235948976631894 |
| 4 | Geographical Zone_kanyamunyu | 0.0009195402298850574 | 2.428796532012334e-05 |
| 5 | Health Facility Name_kasomoro_health_centre | 0.0 | 0.0 |
| 6 | Age | 0.0 | 0.0 |
| 7 | Health Facility Name_kaamuro | 0.0 | 0.0 |
| 8 | Health Facility Name_like | 0.0 | 0.0 |
| 9 | Health Facility Name_kibaire | 0.0 | 0.0 |
| 10 | Health Facility Name_kasomuro | 0.0 | 0.0 |

## Symptom ranking (aggregated permutation importance)

| rank | symptom | aggregate_importance | mean_importance | feature_count |
| --- | --- | --- | --- | --- |
| 1 | Fever (Yes/No) | 0.004801687139262989 | 0.0024008435696314945 | 2 |
| 2 | Headache (Yes/No) | 0.0 | 0.0 | 2 |
| 3 | Chills (Yes/No) | 0.0 | 0.0 | 2 |
| 4 | Vomiting (Yes/No) | 0.0 | 0.0 | 2 |
| 5 | Fatigue (Yes/No) | 0.0 | 0.0 | 2 |
| 6 | Anemia Signs (Yes/No) | 0.0 | 0.0 | 2 |
| 7 | Other Symptoms (Specify) | 0.0 | 0.0 | 20 |

## Artifacts

- Figures: `reports/explainability/figures`
- Tables: `reports/explainability/tables`
