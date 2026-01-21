from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.model import load_model, load_onnx_model, predict_one
from src.api.schemas import PredictRequest, PredictResponse

import warnings

warnings.filterwarnings("ignore")


app = FastAPI(title="Credit Scoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    load_onnx_model()
    #load_model()

@app.get("/")
def root():
    return {
        "message": "Credit Scoring API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred, proba = predict_one(req.model_dump())
    return PredictResponse(prediction=pred, probability=proba)
