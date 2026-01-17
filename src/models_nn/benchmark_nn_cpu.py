from __future__ import annotations

import time
import numpy as np

from src.models_nn.nn_runtime import load_dataset, make_features_array, torch_predict_proba, onnx_predict_proba


# HARD-CODED benchmark params
BATCH_SIZE = 1024
WARMUP = 30
N_RUNS = 200


def time_fn(fn, x: np.ndarray, warmup: int, n_runs: int) -> float:
    for _ in range(warmup):
        _ = fn(x)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = fn(x)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs


def main() -> None:
    df = load_dataset()
    X_arr = make_features_array(df)

    bs = min(BATCH_SIZE, X_arr.shape[0])
    x = X_arr[:bs]

    torch_s = time_fn(torch_predict_proba, x, warmup=WARMUP, n_runs=N_RUNS)
    onnx_s = time_fn(onnx_predict_proba, x, warmup=WARMUP, n_runs=N_RUNS)

    torch_ms_batch = torch_s * 1000
    onnx_ms_batch = onnx_s * 1000

    torch_ms_sample = torch_ms_batch / bs
    onnx_ms_sample = onnx_ms_batch / bs

    torch_rps = bs / torch_s
    onnx_rps = bs / onnx_s

    print("\n=== CPU Inference Benchmark (Torch vs ONNXRuntime) ===")
    print(f"Batch size: {bs}")
    print(f"Torch  : {torch_ms_batch:.3f} ms/batch | {torch_ms_sample:.6f} ms/sample | {torch_rps:.1f} samples/s")
    print(f"ONNXRT : {onnx_ms_batch:.3f} ms/batch | {onnx_ms_sample:.6f} ms/sample | {onnx_rps:.1f} samples/s")
    print(f"Speedup (Torch/ONNXRT): {torch_s / onnx_s:.2f}x")


if __name__ == "__main__":
    main()
