from __future__ import annotations

import os
import joblib
import pandas as pd

from src.data.prepare_data import primary_cleaning, feature_engineering, finalize_dtypes


MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(raw_features: dict):
    model = load_model()

    df = pd.DataFrame([raw_features])

    df = primary_cleaning(df)
    df = feature_engineering(df)
    df = finalize_dtypes(df)

    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= 0.5)
    return pred, proba
