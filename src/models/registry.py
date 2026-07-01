"""Baseline model registry for malaria diagnosis."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _scaled_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _try_register(models: dict[str, Any], name: str, factory: Any) -> None:
    try:
        models[name] = factory()
    except Exception:
        return


def get_baseline_models(random_seed: int = 42) -> dict[str, Any]:
    """Return baseline estimators keyed by model name."""
    models: dict[str, Any] = {
        "logistic_regression": _scaled_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_seed,
            )
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_seed),
        "support_vector_machine": _scaled_pipeline(
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_seed,
            )
        ),
        "mlp": _scaled_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=random_seed,
                early_stopping=True,
            )
        ),
    }

    _try_register(
        models,
        "xgboost",
        lambda: __import__("xgboost", fromlist=["XGBClassifier"]).XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
            scale_pos_weight=6.0,
        ),
    )
    _try_register(
        models,
        "lightgbm",
        lambda: __import__("lightgbm", fromlist=["LGBMClassifier"]).LGBMClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        ),
    )
    _try_register(
        models,
        "catboost",
        lambda: __import__("catboost", fromlist=["CatBoostClassifier"]).CatBoostClassifier(
            iterations=200,
            depth=4,
            random_seed=random_seed,
            verbose=False,
            auto_class_weights="Balanced",
        ),
    )

    return models
