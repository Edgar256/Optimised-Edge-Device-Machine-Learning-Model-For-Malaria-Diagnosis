# Preprocessing Report

Automated preprocessing of the raw Kobo clinical export. **Raw data was not modified.**

## 1. Overview

- **Generated (UTC):** 2026-07-04T08:50:59.540278+00:00
- **Input rows:** 903
- **Output rows:** 844
- **Processed dataset:** `data/processed/processed_dataset.csv`
- **Fitted pipeline:** `data/processed/preprocessing_pipeline.joblib`
- **Encoded feature count:** 71

## 2. Pipeline stages

1. `RowFilterTransformer` — remove duplicate, unlabelled, test, and empty clinical rows.
2. `CategoryNormalizerTransformer` — canonicalize yes/no, gender, season, facility, and zone values.
3. `ImpossibleValueTransformer` — set out-of-range age and fever-duration values to missing.
4. `DateFeatureExtractor` — derive `visit_month` from `Date of Visit`.
5. `ColumnTransformer` — impute, encode categoricals, and optionally scale numeric features.

## 3. Row filtering

| Step | Rows removed |
|------|--------------|
| Duplicate rows | 0 |
| Duplicate `_uuid` | 0 |
| Missing target label | 56 |
| Test patient IDs (Test123) | 1 |
| Empty clinical submissions | 2 |

## 4. Impossible value corrections

- **Age out of range (0–120):** 1 values set to missing
- **Fever duration out of range (0–60 days):** 10 values set to missing

## 5. Missing value handling

- Numeric features (`Age`, `Fever Duration (Days)`): median imputation inside the sklearn pipeline.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- High-cardinality text (`Other Symptoms (Specify)`): normalized to lowercase; one-hot capped at 20 categories.

## 6. Categorical normalization

- Yes/No fields normalized to lowercase `yes` / `no`.
- Gender normalized to `male` / `female`.
- Health facilities mapped to canonical names (`mbaraara`, `kibaire`, `kasomoro_health_centre`, etc.).
- Geographical zones mapped using spelling-variant rules (e.g. `Mbaraar`, `kib aire` → canonical forms).

## 7. Feature encoding and scaling

- **Numeric scaling enabled:** False
- Categorical encoding: `OneHotEncoder(handle_unknown='ignore', max_categories=20)`.
- Post-diagnosis columns (RDT, microscopy, treatment) are excluded to avoid label leakage.

## 8. Target distribution after preprocessing

| Label | Count |
|-------|-------|
| Not malaria | 699 |
| Malaria | 145 |

### Binary target (`target_binary`)

| Value | Count |
|-------|-------|
| 0 | 699 |
| 1 | 145 |

## 9. Output feature names

`Age`, `Fever Duration (Days)`, `Health Facility Name_kaamuro`, `Health Facility Name_kasomoro_health_centre`, `Health Facility Name_kasomuro`, `Health Facility Name_kibaire`, `Health Facility Name_like`, `Health Facility Name_m`, `Health Facility Name_mbaraara`, `Health Facility Name_my`, `Health Facility Name_my favorite`, `Gender (Male/Female)_female`, `Gender (Male/Female)_male`, `Geographical Zone_flue cough headache`, `Geographical Zone_kanyamunyu`, `Geographical Zone_kasamuro`, `Geographical Zone_kasomoro`, `Geographical Zone_kasomuro`, `Geographical Zone_kibaire`, `Geographical Zone_kyabigambire`, `Geographical Zone_like`, `Geographical Zone_m`, `Geographical Zone_mbaraara`, `Geographical Zone_mg`, `Geographical Zone_my`, `Fever (Yes/No)_no`, `Fever (Yes/No)_yes`, `Headache (Yes/No)_no`, `Headache (Yes/No)_yes`, `Chills (Yes/No)_no`, `Chills (Yes/No)_yes`, `Vomiting (Yes/No)_no`, `Vomiting (Yes/No)_yes`, `Fatigue (Yes/No)_no`, `Fatigue (Yes/No)_yes`, `Anemia Signs (Yes/No)_no`, `Anemia Signs (Yes/No)_yes`, `Recent Travel (Yes/No)_no`, `Recent Travel (Yes/No)_yes`, `Exposure Risk (e.g., mosquito-prone area) (Yes/No)_no`, `Exposure Risk (e.g., mosquito-prone area) (Yes/No)_yes`, `Household Malaria History (Yes/No)_no`, `Household Malaria History (Yes/No)_yes`, `Season of Visit (Dry/Rainy)_Dry`, `Season of Visit (Dry/Rainy)_Rainy`, `Other Symptoms (Specify)_cough`, `Other Symptoms (Specify)_cough fever`, `Other Symptoms (Specify)_cough flue`, `Other Symptoms (Specify)_cough flue fever`, `Other Symptoms (Specify)_cough headache`, `Other Symptoms (Specify)_cough headache flue`, `Other Symptoms (Specify)_fever`, `Other Symptoms (Specify)_fever cough`, `Other Symptoms (Specify)_fever cough flue`, `Other Symptoms (Specify)_fever cough headache`, `Other Symptoms (Specify)_fever flue`, `Other Symptoms (Specify)_fever flue cough`, `Other Symptoms (Specify)_fever headache`, `Other Symptoms (Specify)_flue`, `Other Symptoms (Specify)_flue cough`, `Other Symptoms (Specify)_flue fever`, `Other Symptoms (Specify)_headache`, `Other Symptoms (Specify)_headache cough`, `Other Symptoms (Specify)_headache cough flue`, `Other Symptoms (Specify)_infrequent_sklearn`, `visit_month_february`, `visit_month_january`, `visit_month_july`, `visit_month_june`, `visit_month_march`, `visit_month_may`
