from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


RAW_COLUMNS = [
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
    df = pd.read_csv(path)

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw dataset: {sorted(missing)}")

    df = df[RAW_COLUMNS].copy()

    df = df.rename(columns={"default.payment.next.month": "target"})

    return df


def primary_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(keep="first").copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["EDUCATION"].isin([0, 5, 6]), "EDUCATION"] = 4

    df.loc[df["MARRIAGE"].isin([0]), "MARRIAGE"] = 3

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]
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
    out["age_bin"] = pd.cut(
        out["AGE"], bins=bins, labels=labels, right=True, include_lowest=True
    )

    # One-hot encode age_bin -> int8 (0/1)
    age_ohe = pd.get_dummies(
        out["age_bin"],
        prefix="age_bin",
        dummy_na=False,
        dtype=np.int8,
    )

    EXPECTED_AGE_DUMMIES = [
        "age_bin_<=25",
        "age_bin_26-35",
        "age_bin_36-45",
        "age_bin_46-55",
        "age_bin_56-65",
        "age_bin_65+",
    ]

    for col in EXPECTED_AGE_DUMMIES:
        if col not in age_ohe.columns:
            age_ohe[col] = np.int8(0)

    age_ohe = age_ohe[EXPECTED_AGE_DUMMIES]
    out = pd.concat([out.drop(columns=["age_bin"]), age_ohe], axis=1)

    return out


def finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "target" in out.columns:
        out["target"] = (
            pd.to_numeric(out["target"], errors="coerce").fillna(0).astype(np.int8)
        )

    bool_cols = out.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        out[bool_cols] = out[bool_cols].astype(np.int8)

    return out


def prepare_dataset(raw_csv: Path) -> pd.DataFrame:
    df = load_raw_csv(raw_csv)
    df = primary_cleaning(df)
    df = feature_engineering(df)
    df = finalize_dtypes(df)
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, required=True, help="Path to Kaggle CSV")
    parser.add_argument(
        "--out", type=str, required=True, help="Output path (.parquet recommended)"
    )
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
