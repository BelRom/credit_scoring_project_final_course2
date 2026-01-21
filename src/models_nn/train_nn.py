from __future__ import annotations

import warnings
import argparse

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.models.metrics import compute_metrics
from src.models.pipeline import build_preprocessor
from src.models_nn.nn_runtime import TabularMLP, DROPOUT, HIDDEN_SIZES
from src.models_nn.nn_bundle import NNPipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def to_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="target")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)

    args = ap.parse_args()

    df = pd.read_csv(args.data)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int).values

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=57,
        stratify=y,
    )
    pre = build_preprocessor(X_train)
    pre.fit(X_train)

    Xtr = to_float32(pre.transform(X_train))
    Xte = to_float32(pre.transform(X_test))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TabularMLP(in_features=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    bad_epochs = 0

    rng = np.random.default_rng(args.seed)
    n = Xtr.shape[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = rng.permutation(n)

        for start in range(0, n, args.batch_size):
            batch = idx[start : start + args.batch_size]
            xb = torch.from_numpy(Xtr[batch]).to(device)
            yb = torch.from_numpy(y_train[batch].astype(np.float32)).to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits))

        auc = roc_auc_score(y_test, prob)
        print(f"Epoch {epoch:02d} | val ROC-AUC: {auc:.4f}")

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # финальные метрики
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))

    m = compute_metrics(y_test, prob)
    print("\n=== Test metrics (NN) ===")
    print(f"ROC-AUC   : {m.roc_auc:.4f}")
    print(f"Precision : {m.precision:.4f}")
    print(f"Recall    : {m.recall:.4f}")
    print(f"F1-Score  : {m.f1:.4f}")

    # сохранение
    import os

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = NNPipeline(
        preprocessor=pre,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        in_features=Xtr.shape[1],
        device="cpu",  # лучше сохранять cpu, в API сам решишь
    )

    joblib.dump(bundle, f"{args.out_dir}/nn_preprocessor.joblib")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_features": Xtr.shape[1],
            "hidden_sizes": HIDDEN_SIZES,  # если атрибут есть
            "dropout": DROPOUT,
        },
        "models/nn_model.pt",
    )

    print(f"Saved: {args.out_dir}/nn_model.pt")
    print(f"\nSaved: {args.out_dir}/nn_preprocessor.joblib")


if __name__ == "__main__":
    main()
