from __future__ import annotations

from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    top_3_predictions: list[TopPrediction]
    crop: str | None = None
    disease_name: str | None = None
    symptoms: str
    recommended_treatment: str
    severity_level: str | None = None
    mode: str = "mock"


class DiseaseInfo(BaseModel):
    class_name: str
    crop: str | None = None
    disease_name: str | None = None
    symptoms: str
    recommended_treatment: str
    severity_level: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_mode: str
    db_connected: bool


class ScanHistoryItem(BaseModel):
    id: int
    timestamp: str
    predicted_class: str
    confidence: float
    image_hash: str


class FeedbackRequest(BaseModel):
    predicted_class: str
    confidence: float | None = None
    message: str | None = None
