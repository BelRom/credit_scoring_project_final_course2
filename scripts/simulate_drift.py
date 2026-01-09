from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests


def _safe_pct(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pct = arr / max(arr.sum(), 1.0)
    return np.clip(pct, eps, 1.0)


def psi_from_hist(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    eps: float = 1e-6,
) -> float:
    exp_pct = _safe_pct(expected_counts, eps=eps)
    act_pct = _safe_pct(actual_counts, eps=eps)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_numeric(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    strategy: str = "quantile",
    eps: float = 1e-6,
) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if expected.size == 0 or actual.size == 0:
        return float("nan")

    if strategy == "quantile":
        qs = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(expected, qs))

        if edges.size < 3:
            strategy = "uniform"
        else:
            edges[0] = min(edges[0], np.min(actual))
            edges[-1] = max(edges[-1], np.max(actual))

    if strategy == "uniform":
        lo = min(np.min(expected), np.min(actual))
        hi = max(np.max(expected), np.max(actual))
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    return psi_from_hist(exp_counts, act_counts, eps=eps)


def psi_categorical(
    expected: Iterable[Any],
    actual: Iterable[Any],
    eps: float = 1e-6,
) -> float:
    exp_s = pd.Series(list(expected)).astype("str")
    act_s = pd.Series(list(actual)).astype("str")

    exp_counts = exp_s.value_counts(dropna=False)
    act_counts = act_s.value_counts(dropna=False)

    all_cats = exp_counts.index.union(act_counts.index)
    exp_counts = exp_counts.reindex(all_cats, fill_value=0).to_numpy()
    act_counts = act_counts.reindex(all_cats, fill_value=0).to_numpy()

    return psi_from_hist(exp_counts, act_counts, eps=eps)


def call_predict_proba(
    api_url: str,
    rows: List[Dict[str, Any]],
    timeout: float = 10.0,
) -> np.ndarray:
    probs: List[float] = []
    for feat in rows:
        payload = {"features": feat}
        r = requests.post(f"{api_url}/predict", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "probability" not in data:
            raise ValueError(f"API response has no 'probability': {data}")
        probs.append(float(data["probability"]))
    return np.asarray(probs, dtype=float)


@dataclass(frozen=True)
class DriftReport:
    batch_id: int
    n_rows: int
    psi_proba: float
    psi_features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "batch_id": self.batch_id,
            "n_rows": self.n_rows,
            "psi_proba": self.psi_proba,
        }
        for k, v in self.psi_features.items():
            out[f"psi_feat__{k}"] = v
        return out


def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target in df.columns:
        pass
    return df


def split_train_test(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1.0 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(
        drop=True
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simulate new data batches, call API, compute PSI drift."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to processed CSV (contains features + target column).",
    )
    p.add_argument("--target", default="target", help="Target column name in CSV.")
    p.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base url, e.g. http://localhost:8000",
    )
    p.add_argument(
        "--batch-size", type=int, default=200, help="Rows per simulated batch."
    )
    p.add_argument(
        "--batches", type=int, default=5, help="Number of batches to simulate."
    )
    p.add_argument(
        "--sleep", type=float, default=1.0, help="Seconds to sleep between batches."
    )
    p.add_argument("--bins", type=int, default=10, help="Bins for numeric PSI.")
    p.add_argument(
        "--bin-strategy", choices=["quantile", "uniform"], default="quantile"
    )
    p.add_argument(
        "--features",
        default="",
        help="Comma-separated key features to monitor. If empty: auto-select.",
    )
    p.add_argument(
        "--out",
        default="reports/drift_report.csv",
        help="Where to save drift report CSV.",
    )
    args = p.parse_args()

    df = load_data(args.data, args.target)

    all_features = [c for c in df.columns if c != args.target]
    if args.features.strip():
        key_features = [x.strip() for x in args.features.split(",") if x.strip()]
        missing = [c for c in key_features if c not in all_features]
        if missing:
            raise ValueError(f"Features not found in data: {missing}")
    else:
        key_features = all_features[: min(8, len(all_features))]

    train_df, test_df = split_train_test(df, target=args.target, test_size=0.2, seed=42)

    baseline_features = train_df[key_features].copy()

    base_sample = train_df.sample(n=min(len(train_df), 2000), random_state=42)
    base_rows = base_sample[all_features].to_dict(orient="records")
    baseline_probs = call_predict_proba(args.api_url, base_rows)

    reports: List[DriftReport] = []

    n_total = len(test_df)
    if n_total == 0:
        raise ValueError("Test split is empty.")

    cursor = 0
    for b in range(1, args.batches + 1):
        if cursor >= n_total:
            cursor = 0
        batch = test_df.iloc[cursor : cursor + args.batch_size].copy()
        cursor += args.batch_size

        rows = batch[all_features].to_dict(orient="records")
        probs = call_predict_proba(args.api_url, rows)

        psi_p = psi_numeric(
            expected=baseline_probs,
            actual=probs,
            bins=args.bins,
            strategy=args.bin_strategy,
        )

        psi_feats: Dict[str, float] = {}
        for col in key_features:
            if pd.api.types.is_numeric_dtype(baseline_features[col]):
                psi_feats[col] = psi_numeric(
                    expected=baseline_features[col].to_numpy(),
                    actual=batch[col].to_numpy(),
                    bins=args.bins,
                    strategy=args.bin_strategy,
                )
            else:
                psi_feats[col] = psi_categorical(
                    expected=baseline_features[col].astype("str").to_list(),
                    actual=batch[col].astype("str").to_list(),
                )

        rep = DriftReport(
            batch_id=b,
            n_rows=int(len(batch)),
            psi_proba=psi_p,
            psi_features=psi_feats,
        )
        reports.append(rep)

        print(f"\nBatch {b}: n={rep.n_rows}")
        print(f"  PSI(proba): {rep.psi_proba:.6f}")
        for k, v in rep.psi_features.items():
            print(f"  PSI({k}): {v:.6f}")

        time.sleep(max(args.sleep, 0.0))

    out_path = args.out
    out_df = pd.DataFrame([r.to_dict() for r in reports])
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
