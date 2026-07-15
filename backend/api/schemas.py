from __future__ import annotations

from pydantic import BaseModel


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class ModelInputSize(BaseModel):
    width: int
    height: int
    channels: int


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    top_3_predictions: list[TopPrediction]
    crop: str | None = None
    disease_name: str | None = None
    symptoms: str
    recommended_treatment: str
    severity_level: str | None = None
    mode: str = "onnx"
    mock: bool = False
    model_name: str | None = None
    model_version: str | None = None
    input_size: ModelInputSize | None = None


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
    model_name: str | None = None
    model_version: str | None = None
    input_size: ModelInputSize | None = None


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
