# Hyperparameter Optimization Report

- **Generated (UTC):** 2026-07-04T08:51:49.712764+00:00
- **Models optimized:** logistic_regression, decision_tree, catboost
- **Search method:** RandomizedSearchCV (`n_iter=30`, scoring=`roc_auc`)
- **Cross-validation:** Stratified 5-fold

## Leakage prevention

- Preprocessing pipeline is fitted upstream and only transforms features during tuning.
- RandomizedSearchCV operates on preprocessed matrices; labels never enter preprocessing.
- StratifiedKFold preserves class balance within each fold.

## Before optimization

| model_name | roc_auc_mean | roc_auc_ci_low | roc_auc_ci_high | recall_mean | recall_ci_low | recall_ci_high | f1_mean | balanced_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9696062657263352 | 0.9535749108071134 | 0.985637620645557 | 0.9172413793103448 | 0.8591629312828649 | 0.9753198273378246 | 0.9012420253573448 | 0.9464675550200232 |
| decision_tree | 0.9480141404118084 | 0.9263202735317535 | 0.9697080072918633 | 0.9103448275862068 | 0.8666809037505748 | 0.9540087514218388 | 0.919946260635916 | 0.9480141404118086 |
| catboost | 0.9805939681752134 | 0.964124210046039 | 0.9970637263043878 | 0.9448275862068968 | 0.9105747862564482 | 0.9790803861573454 | 0.928868865690205 | 0.96310752383315 |

## After optimization

| model_name | roc_auc_mean | roc_auc_ci_low | roc_auc_ci_high | recall_mean | recall_ci_low | recall_ci_high | f1_mean | balanced_accuracy_mean | best_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9825938264166991 | 0.9664363313141272 | 0.998751321519271 | 0.8827586206896552 | 0.818677499713512 | 0.9468397416657983 | 0.9202902706053069 | 0.9377976042811073 | {'model__C': np.float64(0.04661686413912769), 'model__class_weight': None, 'model__l1_ratio': 1.0, 'model__solver': 'liblinear'} |
| decision_tree | 0.9776232413084311 | 0.9741681904789433 | 0.981078292137919 | 0.9517241379310345 | 0.9174713379805858 | 0.9859769378814832 | 0.9326975154109128 | 0.9665557996952192 | {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 7, 'min_samples_split': 10} |
| catboost | 0.9838090512811426 | 0.9679095080840031 | 0.9997085944782821 | 0.9586206896551724 | 0.9414942896799481 | 0.9757470896303968 | 0.9300158996499563 | 0.9685703653825708 | {'depth': 3, 'iterations': 121, 'l2_leaf_reg': np.float64(1.0635967469774568), 'learning_rate': np.float64(0.01668810326201057)} |

## Best hyperparameters

### `logistic_regression`

```json
{
  "model__C": 0.04661686413912769,
  "model__class_weight": null,
  "model__l1_ratio": 1.0,
  "model__solver": "liblinear"
}
```

### `decision_tree`

```json
{
  "class_weight": "balanced",
  "criterion": "gini",
  "max_depth": 3,
  "min_samples_leaf": 7,
  "min_samples_split": 10
}
```

### `catboost`

```json
{
  "depth": 3,
  "iterations": 121,
  "l2_leaf_reg": 1.0635967469774568,
  "learning_rate": 0.01668810326201057
}
```

## Artifacts

- Before table: `results/hyperparameter_before_optimization.csv`
- After table: `results/hyperparameter_after_optimization.csv`
- Comparison table: `results/hyperparameter_comparison.csv`
- Optimized models: `models/optimized/`
