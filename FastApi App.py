from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from typing import List

# --- SETUP ---
app = FastAPI(title="Predictive Maintenance API", version="1.0")

# Load artifacts
# We use try/except to ensure the app doesn't crash if files aren't generated yet
try:
    model = tf.keras.models.load_model("lstm_rul_model.h5")
    scaler = joblib.load("scaler.pkl")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

# Configuration matches training
SEQUENCE_LENGTH = 50
FEAT_COLS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

# --- Pydantic Models for Input Validation ---

class SensorData(BaseModel):
    # We expect a list of 50 time steps, each containing dictionary of sensors
    # Format: [{"s2": 642.1, "s3": 1580...}, ...]
    history: List[dict] 

class PredictionResponse(BaseModel):
    unit_id: int
    predicted_RUL: float
    status: str

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict_failure(unit_id: int, data: SensorData):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Validation
    if len(data.history) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"Input must contain exactly {SEQUENCE_LENGTH} time steps."
        )

    # 2. Preprocessing
    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(data.history)
        
        # Ensure all required columns exist
        missing_cols = [c for c in FEAT_COLS if c not in df.columns]
        if missing_cols:
             raise HTTPException(status_code=400, detail=f"Missing sensors: {missing_cols}")

        # Select and Scale
        input_data = df[FEAT_COLS].values
        input_scaled = scaler.transform(input_data)
        
        # Reshape for LSTM: (1, 50, 14)
        input_reshaped = input_scaled.reshape(1, SEQUENCE_LENGTH, len(FEAT_COLS))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")

    # 3. Prediction
    prediction = model.predict(input_reshaped)
    rul = float(prediction[0][0])

    # 4. Status Logic
    if rul < 20:
        status = "CRITICAL: Maintenance Required Immediately"
    elif rul < 50:
        status = "WARNING: Plan Maintenance Soon"
    else:
        status = "HEALTHY: Nominal Operation"

    return {
        "unit_id": unit_id,
        "predicted_RUL": round(rul, 2),
        "status": status
    }

# Run with: uvicorn app:app --reload