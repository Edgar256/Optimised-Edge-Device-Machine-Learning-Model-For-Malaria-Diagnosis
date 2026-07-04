# Best Model Explainability Report

- **Model:** `catboost`
- **Generated (UTC):** 2026-07-04T08:52:08.902052+00:00
- **Samples analysed:** 844
- **Features:** 71

## Model performance context

The selected model achieved ROC AUC **0.984** (95% CI [0.968, 1.000]), recall **0.959**, and F1 **0.930** after hyperparameter optimization.

## Symptom influence summary

Based on **permutation importance**, the symptoms most associated with malaria diagnosis are:

- **Other Symptoms (Specify)** (aggregate score 0.0013)
- **Fever (Yes/No)** (aggregate score 0.0012)
- **Headache (Yes/No)** (aggregate score 0.0002)

Positive associations for binary symptom features indicate that reporting the symptom increases predicted malaria risk relative to the reference encoded category.

Based on **SHAP values**, the symptoms most associated with malaria diagnosis are:

- **Fever (Yes/No)** (aggregate score 0.3173)
- **Other Symptoms (Specify)** (aggregate score 0.0286)
- **Headache (Yes/No)** (aggregate score 0.0190)

Positive associations for binary symptom features indicate that reporting the symptom increases predicted malaria risk relative to the reference encoded category.

## Top overall features (permutation importance)

| rank | feature | importance_mean | importance_std |
| --- | --- | --- | --- |
| 1 | Fever Duration (Days) | 0.16326303586404223 | 0.005896696510227466 |
| 2 | Household Malaria History (Yes/No)_yes | 0.007575353953924291 | 0.0018443290043284092 |
| 3 | Household Malaria History (Yes/No)_no | 0.003693453702333355 | 0.0016023943115742992 |
| 4 | Age | 0.0025188693207044643 | 0.00012763819548159924 |
| 5 | Geographical Zone_kasomuro | 0.0021286567016920777 | 0.00047734877987791924 |
| 6 | Health Facility Name_kasomuro | 0.0017300577179221732 | 0.0004073039355915964 |
| 7 | Fever (Yes/No)_no | 0.0011035469389768626 | 0.0014846201677274527 |
| 8 | Other Symptoms (Specify)_fever headache | 0.0005167480637363608 | 0.00012213795259171306 |
| 9 | Other Symptoms (Specify)_fever cough | 0.0003448275862068861 | 0.00010212823895231709 |
| 10 | visit_month_march | 0.00028562971733017695 | 0.00022410811269293253 |

## Symptom ranking (aggregated permutation importance)

| rank | symptom | aggregate_importance | mean_importance | feature_count |
| --- | --- | --- | --- | --- |
| 1 | Other Symptoms (Specify) | 0.0013386118099747992 | 6.693059049873996e-05 | 20 |
| 2 | Fever (Yes/No) | 0.00116126486113165 | 0.000580632430565825 | 2 |
| 3 | Headache (Yes/No) | 0.00024073800009864475 | 0.00012036900004932238 | 2 |
| 4 | Vomiting (Yes/No) | 7.69572295397536e-05 | 3.84786147698768e-05 | 2 |
| 5 | Anemia Signs (Yes/No) | 2.4665778698140794e-06 | 1.2332889349070397e-06 | 2 |
| 6 | Chills (Yes/No) | 9.866311479478362e-07 | 4.933155739739181e-07 | 2 |
| 7 | Fatigue (Yes/No) | -4.933155739628156e-07 | -2.466577869814078e-07 | 2 |

## Artifacts

- Figures: `reports/explainability/figures`
- Tables: `reports/explainability/tables`
