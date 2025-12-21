from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier


SUPPORTED_MODELS = ("logreg", "gboost", "rf", "hgb")


def fix_bool_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """SimpleImputer не любит bool — переводим bool -> int."""
    X = X.copy()
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype("int64")
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Автоматически определяем числовые/категориальные признаки и строим препроцессор."""
    X = fix_bool_dtypes(X)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_estimator(model_name: str, random_state: int = 50):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of: {SUPPORTED_MODELS}")

    if model_name == "logreg":
        # penalty будет тюниться в RandomizedSearchCV
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            random_state=random_state,
        )

    if model_name == "gboost":
        return GradientBoostingClassifier(random_state=random_state)

    if model_name == "rf":
        return RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "hgb":
        return HistGradientBoostingClassifier(random_state=random_state)

    raise ValueError(f"Unknown model_name={model_name}")


def create_pipeline(model_name: str, X: pd.DataFrame, random_state: int = 50) -> Pipeline:
    """Preprocessing + Model."""
    preprocessor = build_preprocessor(X)
    model = build_estimator(model_name, random_state=random_state)
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def get_search_space(model_name: str) -> dict:
    """Гиперпараметры под конкретную модель внутри Pipeline."""
    if model_name == "logreg":
        return {
            "model__C": np.logspace(-3, 2, 30),
            "model__penalty": ["l1", "l2"],
            # Можно добавить elasticnet, но тогда нужен model__l1_ratio и penalty="elasticnet"
        }

    if model_name == "gboost":
        return {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0],
        }

    if model_name == "rf":
        return {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 6, 10, 16],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        }

    if model_name == "hgb":
        return {
            "model__max_iter": [200, 400, 800],
            "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "model__max_depth": [None, 3, 6, 10],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }

    raise ValueError(f"Unknown model_name={model_name}")
