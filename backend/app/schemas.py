from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    date: date
    hour: int = Field(ge=0, le=23)


class PredictionResponse(BaseModel):
    probability: float
    risk_level: str
