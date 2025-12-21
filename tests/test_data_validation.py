import pandas as pd
import pytest
import pandera.pandas as pa


from src.data.validate import validate_raw


def make_minimal_valid_row() -> dict:
    return {
        "ID": 1,
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
        "target": 1,
    }


def test_validation_passes_on_valid_data():
    df = pd.DataFrame([make_minimal_valid_row()])
    validated = validate_raw(df)
    assert len(validated) == 1


def test_validation_fails_on_nulls():
    row = make_minimal_valid_row()
    row["LIMIT_BAL"] = None  # null anomaly
    df = pd.DataFrame([row])

    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


def test_validation_fails_on_out_of_range_values():
    row = make_minimal_valid_row()
    row["AGE"] = 10  # too young
    row["SEX"] = 3   # invalid category
    df = pd.DataFrame([row])

    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)
