# Data Quality Report

Read-only audit of the raw Kobo clinical export. **No data was modified.**

## 1. Dataset overview

- **Source file:** `data/raw/PATIENT_RECORD_DATA_EXTRACTION_FORM.csv`
- **Generated (UTC):** 2026-07-04T08:50:54.810714+00:00
- **Shape:** 903 rows × 37 columns
- **Labelled records:** 847
- **Target prevalence (Malaria):** 17.24%

## 2. All columns

| # | Column | Pandas dtype | Semantic type |
|---|--------|--------------|---------------|
| 1 | `start` | str | datetime |
| 2 | `end` | str | datetime |
| 3 | `Patient ID (Anonymous Code)` | str | text |
| 4 | `Health Facility Name` | str | text |
| 5 | `Date of Visit` | str | datetime |
| 6 | `Age` | float64 | numeric |
| 7 | `Gender (Male/Female)` | str | categorical |
| 8 | `Geographical Zone` | str | text |
| 9 | `Fever (Yes/No)` | str | categorical |
| 10 | `Headache (Yes/No)` | str | categorical |
| 11 | `Chills (Yes/No)` | str | categorical |
| 12 | `Vomiting (Yes/No)` | str | categorical |
| 13 | `Fatigue (Yes/No)` | str | categorical |
| 14 | `Other Symptoms (Specify)` | str | text |
| 15 | `Fever Duration (Days)` | float64 | numeric |
| 16 | `Anemia Signs (Yes/No)` | str | categorical |
| 17 | `Season of Visit (Dry/Rainy)` | str | categorical |
| 18 | `Recent Travel (Yes/No)` | str | categorical |
| 19 | `Exposure Risk (e.g., mosquito-prone area) (Yes/No)` | str | categorical |
| 20 | `Household Malaria History (Yes/No)` | str | categorical |
| 21 | `Rapid Diagnostic Test (RDT) Result (Positive/Negative)` | str | categorical |
| 22 | `Microscopy Result (Positive/Negative)` | str | categorical |
| 23 | `Parasite Density (if available)` | str | text |
| 24 | `Final Confirmed Diagnosis (Malaria / Not Malaria)` | str | categorical |
| 25 | `Treatment Given` | str | text |
| 26 | `Additional Clinical Notes` | float64 | text |
| 27 | `_id` | int64 | integer_identifier |
| 28 | `_uuid` | str | text |
| 29 | `_submission_time` | str | datetime |
| 30 | `_validation_status` | float64 | text |
| 31 | `_notes` | float64 | text |
| 32 | `_status` | str | text |
| 33 | `_submitted_by` | float64 | text |
| 34 | `__version__` | str | text |
| 35 | `_tags` | float64 | text |
| 36 | `meta/rootUuid` | str | text |
| 37 | `_index` | int64 | integer_identifier |

## 3. Detected data types

Pandas inferred types on load. Semantic types reflect intended clinical use.

| Column | Pandas dtype | Semantic type |
|--------|--------------|---------------|
| `start` | str | datetime |
| `end` | str | datetime |
| `Patient ID (Anonymous Code)` | str | text |
| `Health Facility Name` | str | text |
| `Date of Visit` | str | datetime |
| `Age` | float64 | numeric |
| `Gender (Male/Female)` | str | categorical |
| `Geographical Zone` | str | text |
| `Fever (Yes/No)` | str | categorical |
| `Headache (Yes/No)` | str | categorical |
| `Chills (Yes/No)` | str | categorical |
| `Vomiting (Yes/No)` | str | categorical |
| `Fatigue (Yes/No)` | str | categorical |
| `Other Symptoms (Specify)` | str | text |
| `Fever Duration (Days)` | float64 | numeric |
| `Anemia Signs (Yes/No)` | str | categorical |
| `Season of Visit (Dry/Rainy)` | str | categorical |
| `Recent Travel (Yes/No)` | str | categorical |
| `Exposure Risk (e.g., mosquito-prone area) (Yes/No)` | str | categorical |
| `Household Malaria History (Yes/No)` | str | categorical |
| `Rapid Diagnostic Test (RDT) Result (Positive/Negative)` | str | categorical |
| `Microscopy Result (Positive/Negative)` | str | categorical |
| `Parasite Density (if available)` | str | text |
| `Final Confirmed Diagnosis (Malaria / Not Malaria)` | str | categorical |
| `Treatment Given` | str | text |
| `Additional Clinical Notes` | float64 | text |
| `_id` | int64 | integer_identifier |
| `_uuid` | str | text |
| `_submission_time` | str | datetime |
| `_validation_status` | float64 | text |
| `_notes` | float64 | text |
| `_status` | str | text |
| `_submitted_by` | float64 | text |
| `__version__` | str | text |
| `_tags` | float64 | text |
| `meta/rootUuid` | str | text |
| `_index` | int64 | integer_identifier |

## 4. Missing values

| Column | Missing | % Missing |
|--------|---------|-----------|
| `start` | 0 | 0.0% |
| `end` | 0 | 0.0% |
| `Patient ID (Anonymous Code)` | 894 | 99.0% |
| `Health Facility Name` | 60 | 6.64% |
| `Date of Visit` | 57 | 6.31% |
| `Age` | 50 | 5.54% |
| `Gender (Male/Female)` | 54 | 5.98% |
| `Geographical Zone` | 70 | 7.75% |
| `Fever (Yes/No)` | 57 | 6.31% |
| `Headache (Yes/No)` | 65 | 7.2% |
| `Chills (Yes/No)` | 62 | 6.87% |
| `Vomiting (Yes/No)` | 71 | 7.86% |
| `Fatigue (Yes/No)` | 57 | 6.31% |
| `Other Symptoms (Specify)` | 221 | 24.47% |
| `Fever Duration (Days)` | 79 | 8.75% |
| `Anemia Signs (Yes/No)` | 75 | 8.31% |
| `Season of Visit (Dry/Rainy)` | 54 | 5.98% |
| `Recent Travel (Yes/No)` | 59 | 6.53% |
| `Exposure Risk (e.g., mosquito-prone area) (Yes/No)` | 58 | 6.42% |
| `Household Malaria History (Yes/No)` | 60 | 6.64% |
| `Rapid Diagnostic Test (RDT) Result (Positive/Negative)` | 53 | 5.87% |
| `Microscopy Result (Positive/Negative)` | 66 | 7.31% |
| `Parasite Density (if available)` | 900 | 99.67% |
| `Final Confirmed Diagnosis (Malaria / Not Malaria)` | 56 | 6.2% |
| `Treatment Given` | 899 | 99.56% |
| `Additional Clinical Notes` | 903 | 100.0% |
| `_id` | 0 | 0.0% |
| `_uuid` | 0 | 0.0% |
| `_submission_time` | 0 | 0.0% |
| `_validation_status` | 903 | 100.0% |
| `_notes` | 903 | 100.0% |
| `_status` | 0 | 0.0% |
| `_submitted_by` | 903 | 100.0% |
| `__version__` | 0 | 0.0% |
| `_tags` | 903 | 100.0% |
| `meta/rootUuid` | 0 | 0.0% |
| `_index` | 0 | 0.0% |

**High-missing columns (≥30%):** `Patient ID (Anonymous Code)`, `Parasite Density (if available)`, `Treatment Given`, `Additional Clinical Notes`, `_validation_status`, `_notes`, `_submitted_by`, `_tags`

## 5. Duplicate detection

- **Full-row duplicates:** 0
- **Duplicate `_uuid` values:** 0
- **Duplicate patient IDs (including empty):** 894
- **Duplicate non-empty patient IDs:** 1
- **Same patient, same visit date:** 2 rows

Each submission has a unique `_uuid`. One non-empty patient ID (`Kasomoro001`) appears twice on the same date with different ages (10 and 12), suggesting separate visits or inconsistent ID use rather than exact duplicate rows.

## 6. Impossible or suspect values

### 6.1 Numeric fields

**Age**
- Range: 0.0 – 4026.0 years
- Mean / median: 21.96 / 13.0
- Negative ages: 0
- Ages > 120: 1

**Fever Duration (Days)**
- Range: -6.0 – 20.0
- **Negative durations: 10 rows** (impossible; likely data-entry errors)

**Temperature**
- No temperature column exists in this dataset. Checks for values below 30°C or above 45°C are not applicable.

### 6.2 Categorical fields

- **Unexpected diagnosis labels:** None
- **Invalid yes/no values:** None
- **Invalid gender values:** None
- **Invalid season values:** None
- **Invalid RDT values:** None
- **Invalid microscopy values:** None

All yes/no, gender, season, RDT, and microscopy fields use expected vocabularies (case-normalised). Diagnosis labels are limited to `Malaria`, `Not malaria`, or missing.

### 6.3 Suspect response patterns

- **Chills:** 269/283 non-missing entries are `yes` (95.1%). This unusually high rate may reflect default selection in the form.
- **Fatigue:** 270/281 non-missing entries are `yes` (96.1%). Same concern as chills.
- **Geographical zone spelling variants:** `kibaire`, `kibairel`, `kib aire`, `Mbaraar`, `Mbaraarac`, `Mbaraarafl`, `Mnaraara` suggest free-text inconsistency.
- **Health facility naming:** `kibaire` (98) vs `Mbaraara` (174) vs `Mbaraara21` (1) — inconsistent casing and spelling.

## 7. Target variable

**Column:** `Final Confirmed Diagnosis (Malaria / Not Malaria)`

| Label | Count |
|-------|-------|
| Not malaria | 701 |
| Malaria | 146 |
| <MISSING> | 56 |

- **Missing target:** 56 rows (indices: [4, 5, 6, 7, 11, 86, 109, 111, 123, 243, 302, 347, 349, 351, 353, 367, 404, 413, 416, 422, 424, 427, 429, 431, 436, 451, 455, 457, 462, 466, 470, 472, 475, 477, 483, 485, 492, 494, 502, 507, 512, 514, 532, 562, 586, 605, 629, 671, 706, 729, 774, 780, 786, 787, 896, 897])
- **Class imbalance:** 146 Malaria vs 701 Not malaria among labelled rows.

## 8. Cross-field consistency

| Check | Count |
|-------|-------|
| Treatment given without confirmed diagnosis | 2 |
| Malaria diagnosis without recorded treatment | 144 |
| RDT positive but diagnosed Not malaria | 4 |
| RDT negative but diagnosed Malaria | 5 |
| Microscopy positive but diagnosed Not malaria | 6 |
| Microscopy negative but diagnosed Malaria | 10 |

RDT-negative / microscopy-negative cases with a Malaria diagnosis may reflect RDT-first workflows where microscopy was not repeated, or recording timing differences. These rows should be reviewed clinically before modelling.

## 9. Row-level quality flags

- **Empty clinical rows:** 48 (indices: [11, 109, 111, 243, 347, 349, 351, 353, 367, 404, 413, 416, 422, 424, 427, 429, 431, 436, 451, 455, 457, 462, 466, 470, 472, 475, 477, 483, 485, 492, 494, 502, 504, 507, 512, 514, 562, 586, 605, 629, 671, 706, 774, 780, 786, 787, 857, 897])
- **Test record (`Test123`):** 1 row
- **Patient ID missing:** 894 rows (99.0%)

## 10. Summary statistics

Per-column summary statistics are saved to `results/data_quality_summary.csv`.

## 11. Recommendations (audit only — no cleaning applied)

1. Exclude or impute the 10 rows with missing target labels before supervised training.
2. Review 10 rows with negative fever duration; treat as invalid or recode after clinician confirmation.
3. Exclude the test record (`Test123`) and 4 fully empty clinical submissions.
4. Normalise health-facility and geographical-zone text (casing, typos) during preprocessing.
5. Investigate chills/fatigue response patterns — high `yes` rates may reduce feature utility.
6. Plan class-imbalance handling (currently ~13.5% Malaria among labelled rows).
7. Keep RDT/microscopy out of triage-time feature sets to avoid label leakage.
