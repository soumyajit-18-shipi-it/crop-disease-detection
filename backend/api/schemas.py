from __future__ import annotations

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class ModelInputSize(BaseModel):
    width: int
    height: int
    channels: int


class PredictionResponse(BaseModel):
    scan_id: int
    scanned_at: str
    class_name: str
    confidence: float
    top_3_predictions: list[TopPrediction]
    crop: str | None = None
    disease_name: str | None = None
    symptoms: str | None = None
    recommended_treatment: str | None = None
    severity_level: str | None = None
    information_status: str = "unavailable"
    detection_status: str
    quality_status: str
    quality_warnings: list[str] = Field(default_factory=list)
    mode: str = "onnx"
    mock: bool = False
    model_name: str | None = None
    model_version: str | None = None
    input_size: ModelInputSize | None = None


class DiseaseInfo(BaseModel):
    class_name: str
    crop: str | None = None
    disease_name: str | None = None
    symptoms: str | None = None
    recommended_treatment: str | None = None
    severity_level: str | None = None
    information_status: str


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
    original_filename: str | None = None
    content_type: str | None = None
    file_size: int | None = None
    model_name: str | None = None
    model_version: str | None = None
    detection_status: str | None = None
    quality_status: str | None = None
    quality_warnings: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    scan_id: int | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    message: str | None = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    profile_picture: str | None = None
    auth_provider: str
    created_at: str
    last_login_at: str


class SessionResponse(BaseModel):
    authenticated: bool = True
    user: UserResponse
    expires_at: str


class AuthConfigResponse(BaseModel):
    provider: str = "google"
    configured: bool
    callback_url: str | None = None


class DiseaseDistributionItem(BaseModel):
    class_name: str
    count: int
    percentage: float


class DashboardSummary(BaseModel):
    total_scans: int
    healthy_scans: int
    diseased_scans: int
    low_confidence_scans: int
    average_confidence: float | None = None
    healthy_percentage: float | None = None
    active_disease_classes: int
    latest_scan_at: str | None = None
    disease_distribution: list[DiseaseDistributionItem]
    recent_scans: list[ScanHistoryItem]
