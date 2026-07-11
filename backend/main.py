from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.disease_info import router as disease_info_router
from backend.api.routes.predict import router as predict_router


app = FastAPI(title="Crop Disease Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(disease_info_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
