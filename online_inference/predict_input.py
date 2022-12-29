from typing import Literal

from pydantic import BaseModel, validator


class PredictInput(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @classmethod
    def is_valid(cls, value, max_val, name):
        if value > max_val:
            raise ValueError(f'{name} should be in 1-{max_val}]')
        return value

    @validator("age")
    def is_age_valid(cls, value):
        return cls.is_valid(value, 100, "age")

    @validator("trestbps")
    def is_trestbps_valid(cls, value):
        return cls.is_valid(value, 300, "trestbps")

    @validator('chol')
    def is_chol_valid(cls, value):
        return cls.is_valid(value, 500, "chol")

    @validator('thalach')
    def is_thalach_valid(cls, value):
        return cls.is_valid(value, 300, "thalach")

    @validator('oldpeak')
    def is_oldpeak_valid(cls, value):
        return cls.is_valid(value, 10, "oldpeak")