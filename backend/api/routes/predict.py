from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.model_loader import mock_predict
from backend.api.routes.disease_info import get_disease_info_by_class
from backend.api.schemas import PredictionResponse


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        prediction = mock_predict(image_bytes=image_bytes, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    disease = get_disease_info_by_class(str(prediction["class_name"]))
    return PredictionResponse(
        class_name=disease["class_name"],
        confidence=float(prediction["confidence"]),
        symptoms=disease["symptoms"],
        recommended_treatment=disease["recommended_treatment"],
    )
