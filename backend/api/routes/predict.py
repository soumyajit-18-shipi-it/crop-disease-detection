from __future__ import annotations

import hashlib
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.api.model_loader import ModelUnavailableError, model_service
from backend.api.routes.disease_info import db_connect, get_disease_info_by_class
from backend.api.schemas import PredictionResponse
from backend.config import settings


router = APIRouter(tags=["prediction"])


def _validate_upload(content: bytes, content_type: str | None) -> Image.Image:
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_size_mb}MB limit.")
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    try:
        image = Image.open(BytesIO(content))
        image.verify()
        return ImageOps.exif_transpose(Image.open(BytesIO(content))).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc


def _save_scan(class_name: str, confidence: float, image_hash: str) -> None:
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO scans (predicted_class, confidence, image_hash) VALUES (?, ?, ?)",
            (class_name, confidence, image_hash),
        )
        conn.commit()


def _enrich_prediction(prediction: dict, image_hash: str) -> PredictionResponse:
    disease = get_disease_info_by_class(str(prediction["class_name"]))
    _save_scan(str(prediction["class_name"]), float(prediction["confidence"]), image_hash)
    return PredictionResponse(
        class_name=disease["class_name"],
        confidence=float(prediction["confidence"]),
        top_3_predictions=prediction.get("top_3_predictions", []),
        crop=disease.get("crop"),
        disease_name=disease.get("disease_name"),
        symptoms=disease["symptoms"],
        recommended_treatment=disease["recommended_treatment"],
        severity_level=disease.get("severity_level"),
        mode=prediction.get("mode", "onnx"),
        mock=bool(prediction.get("mock", False)),
        model_name=prediction.get("model_name"),
        model_version=prediction.get("model_version"),
        input_size=prediction.get("input_size"),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read()
    image = _validate_upload(content, file.content_type)
    try:
        prediction = model_service.predict(image)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="The inference model is not available.") from exc
    image_hash = hashlib.sha256(content).hexdigest()
    return _enrich_prediction(prediction, image_hash)


@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(files: list[UploadFile] = File(...)) -> list[PredictionResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    responses = []
    for file in files:
        content = await file.read()
        image = _validate_upload(content, file.content_type)
        try:
            prediction = model_service.predict(image)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail="The inference model is not available.") from exc
        responses.append(_enrich_prediction(prediction, hashlib.sha256(content).hexdigest()))
    return responses
