from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort


# -------------------------
# PATHS / PARAMS
# -------------------------
DATA_PATH = Path("data/processed/credit_default.csv")
TARGET_COL = "target"

PREPROCESSOR_PATH = Path("models/nn_preprocessor.joblib")
TORCH_CKPT_PATH = Path("models/nn_model.pt")
ONNX_PATH = Path("models/nn_model.onnx")


# -------------------------
# Model
# -------------------------
class TabularMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers += [nn.Linear(prev, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (N, 1)


# -------------------------
# Checkpoint container
# -------------------------
@dataclass(frozen=True)
class NNCheckpoint:
    in_features: int
    hidden_sizes: tuple[int, ...]
    dropout: float
    state_dict: dict


# -------------------------
# Helpers
# -------------------------
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------
# Data / preprocessing
# -------------------------
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


def make_features_array(df: pd.DataFrame, n_rows: int | None = None) -> np.ndarray:
    pre = load_preprocessor()
    X = df.drop(columns=[TARGET_COL])
    if n_rows is not None:
        X = X.iloc[:n_rows]
    return pre.transform(X).astype(np.float32, copy=False)


# -------------------------
# Torch model
# -------------------------
def load_nn_checkpoint() -> NNCheckpoint:
    ckpt = torch.load(TORCH_CKPT_PATH, map_location="cpu")

    return NNCheckpoint(
        in_features=int(ckpt["in_features"]),
        hidden_sizes=tuple(int(x) for x in ckpt.get("hidden_sizes", (256, 128, 64))),
        dropout=float(ckpt.get("dropout", 0.2)),
        state_dict=ckpt["state_dict"],
    )


def load_torch_model() -> TabularMLP:
    ckpt = load_nn_checkpoint()
    model = TabularMLP(
        in_features=ckpt.in_features,
        hidden_sizes=ckpt.hidden_sizes,
        dropout=ckpt.dropout,
    )
    model.load_state_dict(ckpt.state_dict)
    model.eval()
    return model


def torch_predict_proba(X_arr: np.ndarray) -> np.ndarray:
    model = load_torch_model()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_arr)).cpu().numpy().reshape(-1)
    return sigmoid_np(logits)


# -------------------------
# ONNX runtime
# -------------------------
def load_onnx_session() -> tuple[ort.InferenceSession, str]:
    sess = ort.InferenceSession(
        ONNX_PATH.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def onnx_predict_proba(X_arr: np.ndarray) -> np.ndarray:
    sess, input_name = load_onnx_session()
    logits = sess.run(None, {input_name: X_arr})[0].reshape(-1)
    return sigmoid_np(logits)
