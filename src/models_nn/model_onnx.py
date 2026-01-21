from __future__ import annotations

import os
import numpy as np
import pandas as pd
import onnxruntime as ort


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ONNXModel:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv(
            "MODEL_PATH", "models/nn_model_int8.onnx"
        )

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # в ONNXModel.__init__
        print(
            "input:",
            self.session.get_inputs()[0].name,
            self.session.get_inputs()[0].shape,
            self.session.get_inputs()[0].type,
        )
        print(
            "outputs:", [(o.name, o.shape, o.type) for o in self.session.get_outputs()]
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be pandas.DataFrame, got: {type(X)}")

        x = X.to_numpy(dtype=np.float32)

        y = self.session.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y)

        # Expected logits shape: (N,)
        if y.ndim != 1:
            raise ValueError(
                f"Unexpected ONNX output shape: {y.shape}, expected 1D array"
            )

        p1 = sigmoid(y)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)  # (N,2)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be pandas.DataFrame, got: {type(X)}")

        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)
