from __future__ import annotations

import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema


def raw_input_schema() -> DataFrameSchema:
    """
    Schema for RAW (pre-feature-engineering) dataset.
    Use strict=False to allow extra columns if needed, but we validate required ones.
    """
    return DataFrameSchema(
        columns={
            "ID": Column(int, Check.ge(1), nullable=False),
            "LIMIT_BAL": Column(float, Check.ge(0), nullable=False),
            "SEX": Column(int, Check.isin([1, 2]), nullable=False),
            "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False),
            "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3]), nullable=False),
            "AGE": Column(int, [Check.ge(18), Check.le(120)], nullable=False),

            # Repayment status: typically -2..8 in this dataset (can vary a bit),
            # use a reasonable band.
            "PAY_0": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_2": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_3": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_4": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_5": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),
            "PAY_6": Column(int, [Check.ge(-2), Check.le(10)], nullable=False),

            # Bills and payments should be >= 0 in most cases; bills can be 0.
            "BILL_AMT1": Column(float, nullable=False),
            "BILL_AMT2": Column(float, nullable=False),
            "BILL_AMT3": Column(float, nullable=False),
            "BILL_AMT4": Column(float, nullable=False),
            "BILL_AMT5": Column(float, nullable=False),
            "BILL_AMT6": Column(float, nullable=False),

            "PAY_AMT1": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT2": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT3": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT4": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT5": Column(float, Check.ge(0), nullable=False),
            "PAY_AMT6": Column(float, Check.ge(0), nullable=False),

            # Target must be 0/1
            "target": Column(int, Check.isin([0, 1]), nullable=False),
        },
        strict=False,
        coerce=True,  # coerce types where possible
    )
