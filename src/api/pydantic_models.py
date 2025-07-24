from pydantic import BaseModel, conlist
from typing import List

class CustomerData(BaseModel):
    # Put all features your model expects here as fields
    # Example feature fields (adjust based on your actual features)
    hour: int
    dayofweek: int
    day: int
    month: int
    year: int
    total_amount: float
    avg_amount: float
    transaction_count: float
    std_amount: float
    # Add any categorical feature fields here (must match your preprocessed input)

class PredictionResponse(BaseModel):
    risk_probability: float
