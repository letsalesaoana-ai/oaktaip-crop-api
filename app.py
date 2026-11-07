
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Find the model file beside this application
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "oaktaip_crop_model.pkl"

# Load the trained pipeline
model = joblib.load(MODEL_PATH)

# Create our API
app = FastAPI(
    title="Oaktaip Crop Production Intelligence",
    description="Historical-input cereal production prediction API",
    version="1.0.0"
)


# Define the information the API expects
class PredictionRequest(BaseModel):
    crop: str
    production_lag1: float = Field(ge=0)
    area_planted_lag1: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "company": "Oaktaip Data Intelligence",
        "message": "Crop Production API is running",
        "docs": "/docs"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    crop_lookup = {
        "maize": "Maize",
        "sorghum": "Sorghum",
        "wheat": "Wheat"
    }

    crop_key = request.crop.strip().lower()

    if crop_key not in crop_lookup:
        raise HTTPException(
            status_code=400,
            detail="Crop must be maize, sorghum, or wheat."
        )

    input_data = pd.DataFrame([{
        "Crop": crop_lookup[crop_key],
        "Production_lag1": request.production_lag1,
        "Area_planted_lag1": request.area_planted_lag1
    }])

    prediction = float(model.predict(input_data)[0])

    return {
        "company": "Oaktaip Data Intelligence",
        "crop": crop_lookup[crop_key],
        "predicted_production_mt": round(prediction, 2),
        "unit": "metric tonnes",
        "note": "Model estimate, not a guaranteed harvest."
    }
