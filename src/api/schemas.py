from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    LIMIT_BAL: float = Field(..., ge=0)
    SEX: int = Field(..., ge=1, le=2)
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(..., ge=18, le=120)

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 15000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 3913,
                "BILL_AMT2": 3102,
                "BILL_AMT3": 689,
                "BILL_AMT4": 300,
                "BILL_AMT5": 300,
                "BILL_AMT6": 300,
                "PAY_AMT1": 100,
                "PAY_AMT2": 689,
                "PAY_AMT3": 100,
                "PAY_AMT4": 100,
                "PAY_AMT5": 100,
                "PAY_AMT6": 100,
            }
        }


class PredictResponse(BaseModel):
    prediction: int
    probability: float
