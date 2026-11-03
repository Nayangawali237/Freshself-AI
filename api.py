import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="FreshShelf AI REST Service", version="1.0.0")

MODEL_PATH = os.path.join("models", "freshshelf_pipeline.joblib")

if not os.path.exists(MODEL_PATH):
    import train_model
    train_model.build_and_train()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]

class BatchData(BaseModel):
    produce_type: str = Field(..., example="Tomato")
    initial_maturity: int = Field(..., ge=1, le=5, example=2)
    storage_temp: float = Field(..., ge=-2.0, le=40.0, example=6.5)
    relative_humidity: float = Field(..., ge=10.0, le=100.0, example=85.0)
    ethylene_exposure: str = Field(..., example="Low")
    transit_days: int = Field(..., ge=0, le=30, example=2)
    packaging_type: str = Field(..., example="Perforated Poly")

@app.get("/")
def health_check():
    return {"service": "FreshShelf Prediction Engine", "status": "operational"}

@app.post("/predict")
def predict_shelf_life(item: BatchData):
    try:
        svp = 0.61078 * np.exp((17.27 * item.storage_temp) / (item.storage_temp + 237.3))
        avp = svp * (item.relative_humidity / 100.0)
        vpd = round(svp - avp, 3)
        thermal_stress = max(0.0, item.storage_temp - 8.0)

        df = pd.DataFrame([{
            "produce_type": item.produce_type,
            "initial_maturity": item.initial_maturity,
            "storage_temp": item.storage_temp,
            "relative_humidity": item.relative_humidity,
            "ethylene_exposure": item.ethylene_exposure,
            "transit_days": item.transit_days,
            "packaging_type": item.packaging_type,
            "vpd_kpa": vpd,
            "thermal_stress": thermal_stress
        }])

        pred_days = float(pipeline.predict(df)[0])
        pred_days = round(max(0.1, pred_days), 1)

        if pred_days >= 6.0:
            action, risk = "Full Price Display", "Low"
        elif 3.0 <= pred_days < 6.0:
            action, risk = "Accelerate Rotation", "Medium"
        elif 1.5 <= pred_days < 3.0:
            action, risk = "Trigger 30%-50% Markdown", "High"
        else:
            action, risk = "Repurpose to Processing / Deli", "Critical"

        return {
            "predicted_shelf_life_days": pred_days,
            "risk_band": risk,
            "recommended_action": action,
            "diagnostics": {
                "calculated_vpd_kpa": vpd,
                "thermal_stress_excess_c": thermal_stress
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
