# Crop Disease Detection

AI-powered crop disease detection platform for diagnosing plant leaf diseases from uploaded images. The current project is a working end-to-end skeleton: the backend accepts a leaf image, returns a mock prediction, looks up disease guidance from SQLite, and the frontend displays the result.

The ML model is intentionally mocked until a dataset and trained checkpoint are available.

## Folder Structure

```text
data/                  Dataset storage and class mapping
notebooks/             EDA, preprocessing, training, and evaluation notebooks
src/                   ML training, evaluation, and inference modules
models/                Model configs, checkpoints, and export targets
backend/               FastAPI service and SQLite disease metadata
frontend/              React + Vite UI
docs/                  Project documentation
scripts/               Setup, training, and deployment helper scripts
tests/                 ML unit tests
```

## Run Backend

```bash
cd crop-disease-detection
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/db/seed_disease_data.py
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs will be available at `http://127.0.0.1:8000/docs`.

## Run Frontend

```bash
cd crop-disease-detection/frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually `http://127.0.0.1:5173`.

## Roadmap

- Replace mock inference in `backend/api/model_loader.py` with a real PyTorch checkpoint loader.
- Train transfer learning baselines using `timm` backbones in `src/models/transfer_model.py`.
- Add Grad-CAM visual explanations to inference responses.
- Expand disease metadata and connect recommendations to crop-specific agronomy references.
- Add authentication, scan history persistence, and production deployment configuration.
