from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained joblib model.")
    parser.add_argument("--data", required=True, help="Path to input CSV with features.")
    parser.add_argument("--out", default="data/predictions.csv", help="Output CSV with predictions.")
    parser.add_argument("--proba-threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    X = pd.read_csv(args.data)

    # вероятности
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        # на всякий случай приведём к (0..1) через сигмоиду
        proba = 1 / (1 + np.exp(-scores))

    pred = (proba >= args.proba_threshold).astype(int)

    out_df = X.copy()
    out_df["proba_default"] = proba
    out_df["pred_default"] = pred

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
