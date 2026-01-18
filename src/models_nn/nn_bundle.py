from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class NNPipeline:
    preprocessor: Any  # sklearn ColumnTransformer
    state_dict: Dict[str, torch.Tensor]
    in_features: int
    device: str = "cpu"

    def _build_model(self):
        # импорт внутри, чтобы joblib не ломался из-за циклических импортов
        from src.models_nn.nn_runtime import TabularMLP  # поправь путь на свой
        model = TabularMLP(in_features=self.in_features).to(self.device)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict_proba(self, X):
        # X может быть DataFrame
        Xt = self.preprocessor.transform(X)
        Xt = np.asarray(Xt, dtype=np.float32)

        model = self._build_model()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xt).to(self.device)).cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits))

        # sklearn-style: (n_samples, 2)
        prob = prob.reshape(-1)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X, threshold: float = 0.5):
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(int)

    def transform(self, X):
        return self.preprocessor.transform(X)
