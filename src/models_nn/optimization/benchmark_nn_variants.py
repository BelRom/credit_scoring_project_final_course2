from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import onnxruntime as ort
import warnings

from sklearn.model_selection import train_test_split
from src.models_nn.nn_runtime import TabularMLP

warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path("data/processed/credit_default.csv")
TARGET = "target"
PRE = Path("models/nn_preprocessor.joblib")

TORCH_FP32 = Path("models/nn_model.pt")
TORCH_Q = Path("models/nn_model_quant_dynamic.pt")
ONNX_FP32 = Path("models/nn_model.onnx")
ONNX_INT8 = Path("models/nn_model_int8.onnx")

OUT = Path("reports/bench_optimization.json")

BATCH = 1024
WARMUP = 30
RUNS = 200


def load_batch() -> np.ndarray:
    df = pd.read_csv(DATA)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int).to_numpy()

    _, Xte, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = joblib.load(PRE)
    Xte_arr = pre.transform(Xte).astype(np.float32, copy=False)
    return Xte_arr[: min(BATCH, Xte_arr.shape[0])]


def build_torch_fp32_model(ckpt: dict) -> torch.nn.Module:
    model = TabularMLP(
        in_features=int(ckpt["in_features"]),
        hidden_sizes=tuple(int(v) for v in ckpt.get("hidden_sizes", (256, 128, 64))),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def build_torch_quant_dynamic_model(ckpt: dict) -> torch.nn.Module:
    # 1) FP32 skeleton
    fp32 = TabularMLP(
        in_features=int(ckpt["in_features"]),
        hidden_sizes=tuple(int(v) for v in ckpt.get("hidden_sizes", (256, 128, 64))),
        dropout=float(ckpt.get("dropout", 0.2)),
    )
    fp32.eval()

    # 2) quantize structure
    qmodel = torch.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
    qmodel.eval()

    # 3) load quantized weights
    qmodel.load_state_dict(ckpt["state_dict"])
    qmodel.eval()
    return qmodel


def bench_torch(model: torch.nn.Module, x: np.ndarray, device: str) -> dict:
    model.to(device)
    xt = torch.from_numpy(x).to(device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(xt)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(RUNS):
            _ = model(xt)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    sec = (t1 - t0) / RUNS
    bs = x.shape[0]
    return {
        "ms_per_batch": sec * 1000,
        "ms_per_sample": (sec * 1000) / bs,
        "samples_per_s": bs / sec,
    }


def bench_onnx(onnx_path: Path, x: np.ndarray, providers: list[str]) -> dict:
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=providers)
    name = sess.get_inputs()[0].name

    for _ in range(WARMUP):
        _ = sess.run(None, {name: x})

    t0 = time.perf_counter()
    for _ in range(RUNS):
        _ = sess.run(None, {name: x})
    t1 = time.perf_counter()

    sec = (t1 - t0) / RUNS
    bs = x.shape[0]
    return {
        "ms_per_batch": sec * 1000,
        "ms_per_sample": (sec * 1000) / bs,
        "samples_per_s": bs / sec,
    }


def main() -> None:
    x = load_batch()
    bs = int(x.shape[0])

    ckpt_fp32 = torch.load(TORCH_FP32.as_posix(), map_location="cpu")
    ckpt_q = torch.load(TORCH_Q.as_posix(), map_location="cpu")

    model_fp32 = build_torch_fp32_model(ckpt_fp32)
    model_q = build_torch_quant_dynamic_model(ckpt_q)

    res: dict = {"batch_size": bs}

    # CPU
    res["cpu"] = {
        "torch_fp32": bench_torch(model_fp32, x, "cpu"),
        "torch_quant_dynamic": bench_torch(model_q, x, "cpu"),
        "onnx_fp32": bench_onnx(ONNX_FP32, x, ["CPUExecutionProvider"]),
        "onnx_int8": bench_onnx(ONNX_INT8, x, ["CPUExecutionProvider"]),
    }

    # GPU (если доступен torch cuda)
    if torch.cuda.is_available():
        gpu = {
            "torch_fp32": bench_torch(model_fp32, x, "cuda"),
        }

        # ONNX GPU provider (если установлен onnxruntime-gpu)
        try:
            gpu["onnx_fp32"] = bench_onnx(
                ONNX_FP32, x, ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            gpu["onnx_int8"] = bench_onnx(
                ONNX_INT8, x, ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception:
            gpu["onnx"] = "onnxruntime-gpu / CUDAExecutionProvider not available"
        res["gpu"] = gpu
    else:
        res["gpu"] = "torch cuda not available"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
