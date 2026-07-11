from pydantic import BaseModel


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    symptoms: str
    recommended_treatment: str


class DiseaseInfo(BaseModel):
    class_name: str
    symptoms: str
    recommended_treatment: str
