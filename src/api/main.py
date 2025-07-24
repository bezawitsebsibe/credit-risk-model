from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load the best model from MLflow Model Registry (adjust the model name and version)
MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"  # or "Staging"

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert Pydantic model to DataFrame or array
    input_data = customer.dict()
    input_array = np.array([list(input_data.values())])

    # Predict risk probability
    pred_proba = model.predict(input_array)

    # If your model returns class probabilities, select the probability for the positive class
    if hasattr(pred_proba, 'tolist'):
        pred_proba = pred_proba.tolist()
    risk_probability = float(pred_proba[0])  # Adjust if your model returns probabilities differently

    return PredictionResponse(risk_probability=risk_probability)
