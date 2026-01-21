from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import onnxruntime as ort
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.models_nn.nn_runtime import TabularMLP

warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path("data/processed/credit_default.csv")
TARGET = "target"

PRE = Path("models/nn_preprocessor.joblib")

TORCH_FP32 = Path("models/nn_model.pt")
TORCH_Q = Path("models/nn_model_quant_dynamic.pt")
ONNX_FP32 = Path("models/nn_model.onnx")
ONNX_INT8 = Path("models/nn_model_int8.onnx")

OUT = Path("reports/metrics_optimization.json")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_test() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int).to_numpy()

    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = joblib.load(PRE)
    Xte_arr = pre.transform(Xte).astype(np.float32, copy=False)
    return Xte_arr, yte


def torch_fp32_proba(pt_path: Path, X: np.ndarray) -> np.ndarray:
    ckpt = torch.load(pt_path.as_posix(), map_location="cpu")

    model = TabularMLP(
        in_features=int(ckpt["in_features"]),
        hidden_sizes=tuple(int(v) for v in ckpt.get("hidden_sizes", (256, 128, 64))),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X)).cpu().numpy().reshape(-1)

    return sigmoid(logits)


def torch_quant_dynamic_proba(pt_path: Path, X: np.ndarray) -> np.ndarray:
    ckpt = torch.load(pt_path.as_posix(), map_location="cpu")

    # 1) собрать FP32 модель (архитектура нужна)
    fp32 = TabularMLP(
        in_features=int(ckpt["in_features"]),
        hidden_sizes=tuple(int(v) for v in ckpt.get("hidden_sizes", (256, 128, 64))),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    fp32.eval()

    # 2) применить dynamic quantization (Linear -> int8)
    qmodel = torch.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
    qmodel.eval()

    # 3) загрузить quantized state_dict (packed_params/scale/zero_point)
    qmodel.load_state_dict(ckpt["state_dict"])

    with torch.no_grad():
        logits = qmodel(torch.from_numpy(X)).cpu().numpy().reshape(-1)

    return sigmoid(logits)


def onnx_proba(onnx_path: Path, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name
    logits = sess.run(None, {name: X})[0].reshape(-1)
    return sigmoid(logits)


def main() -> None:
    X, y = load_test()

    res = {
        "torch_fp32": metrics(y, torch_fp32_proba(TORCH_FP32, X)),
        "torch_quant_dynamic": metrics(y, torch_quant_dynamic_proba(TORCH_Q, X)),
        "onnx_fp32": metrics(y, onnx_proba(ONNX_FP32, X)),
        "onnx_int8": metrics(y, onnx_proba(ONNX_INT8, X)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
