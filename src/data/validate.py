from __future__ import annotations

import os
import warnings

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
warnings.filterwarnings(
    "ignore",
    message=r"Importing pandas-specific classes and functions from the top-level pandera module.*",
    category=FutureWarning,
)

from pathlib import Path
import pandas as pd
import pandera.pandas as pa

from .validation_schema import raw_input_schema


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates raw dataset. Raises pandera.errors.SchemaError on failure.
    Returns validated (and type-coerced) dataframe.
    """
    schema = raw_input_schema()
    return schema.validate(df, lazy=True)  # lazy=True collects all errors


def validate_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # rename target if needed
    if "default.payment.next.month" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"default.payment.next.month": "target"})
    return validate_raw(df)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", type=str, required=True)
    args = parser.parse_args()

    validated = validate_raw_csv(Path(args.raw_csv))
    print(f"Validation OK: shape={validated.shape}")


if __name__ == "__main__":
    main()
