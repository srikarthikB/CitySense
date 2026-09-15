from fastapi import FastAPI, HTTPException
from app.features import create_inference_row
from app.model import predict_probability
from app.schemas import PredictionRequest, PredictionResponse
from fastapi import FastAPI, HTTPException

app = FastAPI(title="CitySense API")

@app.get("/")
def root():
    return {"message": "CitySense API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not (40.49 <= request.latitude <= 40.91 and -74.26 <= request.longitude <= -73.71):
        raise HTTPException(status_code=400, detail="Location is outside the supported CitySense area")
    features = create_inference_row(
        request.latitude,
        request.longitude,
        request.date,
        request.hour
    )
    probability = predict_probability(features)
    if probability < 0.10:
        risk_level = "Low"
    elif probability <= 0.20:
        risk_level = "Medium"
    else:
        risk_level = "High"
    return PredictionResponse(
        probability=probability,
        risk_level=risk_level
    )