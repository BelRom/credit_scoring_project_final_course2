from __future__ import annotations

import os
import joblib
import pandas as pd

from src.data.prepare_data import primary_cleaning, feature_engineering, finalize_dtypes
from src.models_nn.model_onnx import ONNXModel

#MODEL_PATH = os.getenv("MODEL_PATH", "models/nn_model.onnx")

MODEL_PATH = os.getenv("MODEL_PATH", "models/nn_model_int8.onnx")
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def load_onnx_model() -> ONNXModel:
    global _model
    if _model is None:
        _model = ONNXModel(MODEL_PATH)
    return _model


def predict_one(raw_features: dict):
    model = load_onnx_model()
    #model = load_model()
    features = raw_features.get("features", raw_features)
    df = pd.DataFrame([features])
    df = primary_cleaning(df)
    df = feature_engineering(df)
    df = finalize_dtypes(df)

    print("df shape:", df.shape)
    print("df sum abs:", float(df.abs().sum(axis=1).iloc[0]))
    print("first 10 cols:", df.columns.tolist()[:10])
    print("first row first 10:", df.iloc[0, :10].to_list())

    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= 0.5)

    return pred, proba
