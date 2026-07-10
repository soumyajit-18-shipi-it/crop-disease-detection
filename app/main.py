from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.model import CropDiseaseModel

app = FastAPI(
    title="Crop Disease Detection API",
    description="An AI-powered system that uses computer vision to detect crop diseases from plant images.",
    version="0.1.0",
)

model = CropDiseaseModel()


@app.get("/")
def read_root() -> JSONResponse:
    """API health status check endpoint."""
    return JSONResponse(content={"status": "running", "service": "Crop Disease Detection API"})


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> JSONResponse:
    """Predict crop disease from uploaded leaf image."""
    if not file.filename:
        return JSONResponse(
            status_code=400, content={"error": "Uploaded file must have a filename."}
        )

    prediction_result = model.predict(file.filename)
    return JSONResponse(content=prediction_result)
