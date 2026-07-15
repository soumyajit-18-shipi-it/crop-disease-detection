from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.api.auth import AuthContext, require_csrf
from backend.api.model_loader import ModelUnavailableError, model_service
from backend.api.routes.disease_info import db_connect, get_disease_info_by_class
from backend.api.schemas import PredictionResponse
from backend.config import settings
from backend.db.database import database_json, timestamp_string


router = APIRouter(tags=["prediction"])
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_BATCH_FILES = 10
Image.MAX_IMAGE_PIXELS = 40_000_000


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read(settings.max_upload_size_bytes + 1)
    await file.close()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_size_mb}MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    return content


def _validate_upload(content: bytes, content_type: str | None) -> Image.Image:
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_size_mb}MB limit.")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Supported image formats are JPEG, PNG, and WebP.")
    try:
        probe = Image.open(BytesIO(content))
        image_format = probe.format
        probe.verify()
        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise HTTPException(status_code=415, detail="Supported image formats are JPEG, PNG, and WebP.")
        image = ImageOps.exif_transpose(Image.open(BytesIO(content))).convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc
    if min(image.size) < 64:
        raise HTTPException(status_code=422, detail="Image is too small. Both dimensions must be at least 64 pixels.")
    if image.width * image.height > int(Image.MAX_IMAGE_PIXELS or 40_000_000):
        raise HTTPException(status_code=413, detail="Image dimensions exceed the 40-megapixel safety limit.")
    return image


def _assess_image_quality(image: Image.Image) -> tuple[str, list[str]]:
    sample = np.asarray(image.resize((256, 256)), dtype=np.uint8)
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if contrast < 2.0:
        raise HTTPException(status_code=422, detail="Image has insufficient visual detail for a reliable scan.")
    warnings: list[str] = []
    if mean_brightness < 25:
        warnings.append("Image is very dark; retake it in indirect natural light.")
    elif mean_brightness > 235:
        warnings.append("Image is overexposed; reduce glare and retake it.")
    if contrast < 18:
        warnings.append("Image has low contrast; center the leaf against a plain background.")
    if sharpness < 12:
        warnings.append("Image may be blurred; hold the camera steady and retake it.")
    return ("review_recommended" if warnings else "acceptable"), warnings


def _detection_status(class_name: str, confidence: float, quality_status: str) -> str:
    if confidence < settings.low_confidence_threshold:
        return "low_confidence"
    if quality_status != "acceptable":
        return "review_recommended"
    if "healthy" in class_name.lower():
        return "healthy"
    return "disease_detected"


def _save_scan(
    *,
    auth: AuthContext,
    class_name: str,
    confidence: float,
    image_hash: str,
    original_filename: str | None,
    content_type: str | None,
    file_size: int,
    model_name: str | None,
    model_version: str | None,
    detection_status: str,
    quality_status: str,
    quality_warnings: list[str],
) -> tuple[int, str]:
    safe_filename = Path(original_filename or "upload").name[:255]
    with db_connect() as connection:
        row = connection.execute(
            """
            INSERT INTO scans(
                user_id, predicted_class, confidence, image_hash, original_filename,
                content_type, file_size, model_name, model_version, detection_status,
                quality_status, quality_warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id, timestamp
            """,
            (
                auth.user_id,
                class_name,
                confidence,
                image_hash,
                safe_filename,
                content_type,
                file_size,
                model_name,
                model_version,
                detection_status,
                quality_status,
                database_json(quality_warnings),
            ),
        ).fetchone()
        scan_id = int(row["id"])
        scanned_at = timestamp_string(row["timestamp"]) or ""
        connection.commit()
    return scan_id, scanned_at


def _enrich_prediction(
    prediction: dict,
    *,
    content: bytes,
    filename: str | None,
    content_type: str | None,
    quality_status: str,
    quality_warnings: list[str],
    auth: AuthContext,
) -> PredictionResponse:
    class_name = str(prediction["class_name"])
    confidence = float(prediction["confidence"])
    disease = get_disease_info_by_class(class_name)
    detection_status = _detection_status(class_name, confidence, quality_status)
    scan_id, scanned_at = _save_scan(
        auth=auth,
        class_name=class_name,
        confidence=confidence,
        image_hash=hashlib.sha256(content).hexdigest(),
        original_filename=filename,
        content_type=content_type,
        file_size=len(content),
        model_name=prediction.get("model_name"),
        model_version=prediction.get("model_version"),
        detection_status=detection_status,
        quality_status=quality_status,
        quality_warnings=quality_warnings,
    )
    return PredictionResponse(
        scan_id=scan_id,
        scanned_at=scanned_at,
        class_name=class_name,
        confidence=confidence,
        top_3_predictions=prediction.get("top_3_predictions", []),
        crop=disease.get("crop"),
        disease_name=disease.get("disease_name"),
        symptoms=disease.get("symptoms"),
        recommended_treatment=disease.get("recommended_treatment"),
        severity_level=disease.get("severity_level"),
        information_status=disease.get("information_status", "unavailable"),
        detection_status=detection_status,
        quality_status=quality_status,
        quality_warnings=quality_warnings,
        mode=prediction.get("mode", "onnx"),
        mock=bool(prediction.get("mock", False)),
        model_name=prediction.get("model_name"),
        model_version=prediction.get("model_version"),
        input_size=prediction.get("input_size"),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    auth: AuthContext = Depends(require_csrf),
) -> PredictionResponse:
    content = await _read_upload(file)
    image = _validate_upload(content, file.content_type)
    quality_status, quality_warnings = _assess_image_quality(image)
    try:
        prediction = model_service.predict(image)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="The inference model is not available.") from exc
    return _enrich_prediction(
        prediction,
        content=content,
        filename=file.filename,
        content_type=file.content_type,
        quality_status=quality_status,
        quality_warnings=quality_warnings,
        auth=auth,
    )


@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    files: list[UploadFile] = File(...),
    auth: AuthContext = Depends(require_csrf),
) -> list[PredictionResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"A batch can contain at most {MAX_BATCH_FILES} images.")
    responses: list[PredictionResponse] = []
    for file in files:
        content = await _read_upload(file)
        image = _validate_upload(content, file.content_type)
        quality_status, quality_warnings = _assess_image_quality(image)
        try:
            prediction = model_service.predict(image)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail="The inference model is not available.") from exc
        responses.append(
            _enrich_prediction(
                prediction,
                content=content,
                filename=file.filename,
                content_type=file.content_type,
                quality_status=quality_status,
                quality_warnings=quality_warnings,
                auth=auth,
            )
        )
    return responses
