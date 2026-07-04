# Feature Engineering Report

Triage-time clinical features derived from the cleaned processed dataset. **Post-diagnosis columns are excluded to prevent target leakage.**

## 1. Overview

- **Generated (UTC):** 2026-07-04T08:51:03.572714+00:00
- **Input rows:** 844
- **Output rows:** 844
- **Engineered features created:** 12
- **Output dataset:** `data/processed/engineered_dataset.csv`

## 2. Feature catalogue

| Feature | Type | Definition | Medical rationale |
|---------|------|------------|-------------------|
| `age_group` | categorical | child (<18), adult (18-59), senior (>=60) | Paediatric and elderly patients differ in immune response, malaria presentation, and complication risk; age stratification is standard in Uganda health records. |
| `pediatric_high_risk` | binary | 1 if age <= 4 years, else 0 | WHO prioritises under-5 malaria surveillance in endemic regions because young children face the highest morbidity and mortality burden. |
| `fever_severity` | categorical | none (no fever), acute (fever 0-2 days), prolonged (fever >=3 days) | No body-temperature column exists; fever presence and duration proxy febrile severity. Prolonged fever raises clinical suspicion of malaria or complications. |
| `symptom_count` | numeric | Count of yes responses across fever, headache, chills, vomiting, fatigue | Poly-symptom presentations are associated with higher malaria probability at triage. |
| `core_symptom_count` | numeric | Count of yes across fever, headache, vomiting only | Excludes chills and fatigue which exceed 95% yes in this dataset and add little discriminative signal, likely due to form-default selection. |
| `transmission_risk_score` | numeric | Sum (0-4) of exposure risk, household malaria history, recent travel, rainy season | Combines entomological exposure, household clustering, travel, and seasonal transmission into a single interpretable endemic-risk index. |
| `is_rainy_season` | binary | 1 if season is Rainy, else 0 | Rainy season increases mosquito breeding and malaria transmission in Hoima District. |
| `has_other_symptoms` | binary | 1 if free-text other symptoms field is non-empty | Captures additional complaints (cough, diarrhoea, etc.) not covered by the structured yes/no symptom checklist. |
| `exposure_x_rainy_season` | numeric | exposure_risk_flag * is_rainy_season | Mosquito-prone exposure during rainy season multiplies vector contact and transmission risk beyond either factor alone. |
| `household_history_x_child` | numeric | household_malaria_history_flag * is_child_flag | Children in households with recent malaria face elevated reinfection risk due to shared exposure within the home. |
| `anemia_signs_x_pediatric` | numeric | anemia_signs_flag * pediatric_high_risk | Anaemia signs in children under five are a severity marker in malaria-endemic areas; hemoglobin is not recorded so clinical anaemia signs are used as proxy. |
| `fever_x_duration` | numeric | fever_flag * fever_duration_days (NaN when duration missing) | Combines fever presence with persistence; sustained febrile illness is more suggestive of malaria than isolated fever reporting without duration. |

## 3. Excluded features (leakage prevention)

The following features were **not** created because they rely on post-diagnosis information or unavailable measurements:

| Feature | Reason excluded |
|---------|-----------------|
| Fever x RDT | RDT is a diagnostic test result recorded during workup, not available at triage. |
| Microscopy x RDT | Both microscopy and RDT are post-diagnosis columns that would leak the target. |
| Hemoglobin x Age | No hemoglobin column exists; replaced by anemia_signs_x_pediatric interaction. |
| Temperature severity (Normal/Moderate/High) | No body-temperature column exists; replaced by fever_severity proxy. |

Post-diagnosis columns blocked by configuration:

`Rapid Diagnostic Test (RDT) Result (Positive/Negative)`, `Microscopy Result (Positive/Negative)`, `Parasite Density (if available)`, `Treatment Given`, `Additional Clinical Notes`

## 4. Feature distributions

### `age_group`

- `child`: 506
- `adult`: 322
- `senior`: 13
- `nan`: 3

### `pediatric_high_risk`

- `0`: 647
- `1`: 197

### `fever_severity`

- `none`: 490
- `acute`: 352
- `prolonged`: 2

### `symptom_count`

- `2`: 405
- `3`: 295
- `4`: 112
- `1`: 20
- `5`: 10
- `0`: 2

### `core_symptom_count`

- `0`: 408
- `1`: 306
- `2`: 120
- `3`: 10

### `transmission_risk_score`

- `2`: 698
- `3`: 120
- `1`: 23
- `0`: 3

### `is_rainy_season`

- `1`: 830
- `0`: 14

### `has_other_symptoms`

- `1`: 675
- `0`: 169

### `exposure_x_rainy_season`

- `1`: 814
- `0`: 30

### `household_history_x_child`

- `0`: 761
- `1`: 83

### `anemia_signs_x_pediatric`

- `0`: 840
- `1`: 4

### `fever_x_duration`

- `0.0`: 686
- `2.0`: 117
- `nan`: 39
- `3.0`: 1
- `20.0`: 1

## 5. Data quality caveats

- **Chills and fatigue** are excluded from `core_symptom_count` because both exceed 95% `yes` in the source data, suggesting form-default selection rather than discriminative clinical signal.
- **Fever duration** may contain imputed values from the preprocessing pipeline where negative durations were corrected to missing before median imputation.
- **No body-temperature or hemoglobin** measurements exist; `fever_severity` and `anemia_signs_x_pediatric` use the closest available clinical proxies.
- **RDT and microscopy** are intentionally excluded; interaction features involving these tests would encode diagnostic outcomes rather than triage-time presentation.
