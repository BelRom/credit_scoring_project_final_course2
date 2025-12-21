from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default.payment.next.month",
]


@dataclass(frozen=True)
class DataPaths:
    raw_csv: Path
    prepared_parquet: Path


def load_raw_csv(path: Path) -> pd.DataFrame:
    """
    Load Kaggle CSV and normalize column names/target column.
    Kaggle versions sometimes contain a header row with different naming,
    but you specified the expected column names explicitly.
    """
    df = pd.read_csv(path)

    # If file contains unexpected column names, try to map them to expected ones
    # (optional safety). Otherwise ensure required columns exist.
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw dataset: {sorted(missing)}")

    # Keep only expected columns in the correct order
    df = df[RAW_COLUMNS].copy()

    # Rename target to a cleaner name
    df = df.rename(columns={"default.payment.next.month": "target"})

    return df


def primary_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary cleaning:
    - drop duplicates by ID (keep first)
    - enforce numeric dtypes where possible
    - basic category normalization for known 'unknown' values
    """
    df = df.drop_duplicates(subset=["ID"], keep="first").copy()

    # Convert everything numeric (dataset is numeric-coded)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize known odd codes (typical for this dataset):
    # EDUCATION: 0, 5, 6 often mean "others/unknown" in many implementations
    df.loc[df["EDUCATION"].isin([0, 5, 6]), "EDUCATION"] = 4  # 4 = others

    # MARRIAGE: 0 often means "others/unknown"
    df.loc[df["MARRIAGE"].isin([0]), "MARRIAGE"] = 3  # 3 = others

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    out = df.copy()

    # Repayment behavior aggregates
    out["pay_mean"] = out[pay_cols].mean(axis=1)
    out["pay_max"] = out[pay_cols].max(axis=1)
    out["pay_min"] = out[pay_cols].min(axis=1)
    out["pay_std"] = out[pay_cols].std(axis=1).fillna(0.0)
    out["num_delays"] = (out[pay_cols] > 0).sum(axis=1)

    # Bills/payments aggregates
    out["bill_sum"] = out[bill_cols].sum(axis=1)
    out["bill_mean"] = out[bill_cols].mean(axis=1)
    out["pay_amt_sum"] = out[amt_cols].sum(axis=1)
    out["pay_amt_mean"] = out[amt_cols].mean(axis=1)

    # Ratios (safe divide)
    eps = 1e-6
    out["utilization_ratio"] = out["bill_mean"] / (out["LIMIT_BAL"] + eps)
    out["payment_to_bill_ratio"] = out["pay_amt_sum"] / (out["bill_sum"].abs() + eps)

    # Age binning
    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]
    out["age_bin"] = pd.cut(out["AGE"], bins=bins, labels=labels, right=True, include_lowest=True)

    # One-hot encode age_bin -> int8 (0/1)
    age_ohe = pd.get_dummies(
        out["age_bin"],
        prefix="age_bin",
        dummy_na=False,
        dtype=np.int8,  # <-- ключевая правка
    )
    out = pd.concat([out.drop(columns=["age_bin"]), age_ohe], axis=1)

    return out


def finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final dtype normalization to avoid bool dtypes and keep dataset compact & sklearn-friendly.
    """
    out = df.copy()

    # target -> int8
    if "target" in out.columns:
        out["target"] = pd.to_numeric(out["target"], errors="coerce").fillna(0).astype(np.int8)

    # any bool -> int8 (safety net)
    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(np.int8)

    return out


def prepare_dataset(raw_csv: Path) -> pd.DataFrame:
    df = load_raw_csv(raw_csv)
    df = primary_cleaning(df)
    df = feature_engineering(df)
    df = finalize_dtypes(df)  # ✅ добавили финализацию типов
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, required=True, help="Path to Kaggle CSV")
    parser.add_argument("--out", type=str, required=True, help="Output path (.parquet recommended)")
    args = parser.parse_args()

    df_prepared = prepare_dataset(Path(args.raw_csv))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df_prepared.to_parquet(out_path, index=False)
    else:
        df_prepared.to_csv(out_path, index=False)

    print(f"Prepared dataset saved to: {out_path} | shape={df_prepared.shape}")


if __name__ == "__main__":
    main()
