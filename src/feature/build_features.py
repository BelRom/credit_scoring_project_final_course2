from __future__ import annotations
import numpy as np
import pandas as pd

from src.data.schemas import raw_predict_schema
from src.data.prepare_data import feature_engineering, finalize_dtypes, primary_cleaning

RAW_FEATURE_COLUMNS = [
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
]


def prepare_features_for_inference(raw_row: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_row])

    # pandera валидация
    df = raw_predict_schema().validate(df)

    df = primary_cleaning(df)
    df = feature_engineering(df)
    df = finalize_dtypes(df)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pay_amts = df[
        ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    ]
    bill_amts = df[
        ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    ]
    delays = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]

    df["pay_mean"] = pay_amts.mean(axis=1)
    df["pay_std"] = pay_amts.std(axis=1)
    df["pay_min"] = pay_amts.min(axis=1)
    df["pay_max"] = pay_amts.max(axis=1)
    df["pay_amt_sum"] = pay_amts.sum(axis=1)

    df["bill_mean"] = bill_amts.mean(axis=1)
    df["bill_sum"] = bill_amts.sum(axis=1)

    df["payment_to_bill_ratio"] = df["pay_amt_sum"] / (
        df["bill_sum"].replace(0, np.nan)
    )
    df["payment_to_bill_ratio"] = df["payment_to_bill_ratio"].fillna(0.0)

    df["utilization_ratio"] = df["bill_mean"] / (df["LIMIT_BAL"].replace(0, np.nan))
    df["utilization_ratio"] = df["utilization_ratio"].fillna(0.0)

    df["num_delays"] = (delays > 0).sum(axis=1)

    df["age_bin"] = pd.cut(
        df["AGE"],
        bins=[0, 25, 35, 45, 55, 65, 200],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
        right=True,
        include_lowest=True,
    )

    # one-hot
    age_dummies = pd.get_dummies(df["age_bin"], prefix="age_bin")
    df = pd.concat([df.drop(columns=["age_bin"]), age_dummies], axis=1)

    return df
