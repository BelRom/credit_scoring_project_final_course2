from __future__ import annotations

import numpy as np

from src.models_nn.nn_runtime import load_dataset, make_features_array, torch_predict_proba, onnx_predict_proba


# HARD-CODED validation params
N_SAMPLES = 2048
RTOL = 1e-4
ATOL = 1e-5


def main() -> None:
    df = load_dataset()
    X_arr = make_features_array(df, n_rows=N_SAMPLES)

    p_torch = torch_predict_proba(X_arr)
    p_onnx = onnx_predict_proba(X_arr)

    abs_diff = float(np.max(np.abs(p_torch - p_onnx)))
    rel_diff = float(np.max(np.abs(p_torch - p_onnx) / (np.abs(p_torch) + 1e-12)))
    ok = bool(np.allclose(p_torch, p_onnx, rtol=RTOL, atol=ATOL))

    print("\n=== ONNX conversion validation ===")
    print(f"Samples: {len(p_torch)}")
    print(f"max(abs(prob_torch - prob_onnx)) = {abs_diff:.8f}")
    print(f"max(rel diff)                   = {rel_diff:.8f}")
    print(f"allclose(rtol={RTOL}, atol={ATOL}) -> {ok}")

    if not ok:
        raise SystemExit("❌ Validation FAILED: outputs differ beyond tolerance.")
    print("✅ Validation PASSED.")


if __name__ == "__main__":
    main()
