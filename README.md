# Crop Disease Detection

Production-oriented crop disease detection platform for classifying plant leaf diseases from images. The system includes a real data pipeline, PyTorch/timm training code, ONNX export for lighter serving, a FastAPI backend, SQLite disease metadata/history, and a React + Vite frontend.

The backend is designed to run in two modes:

- `onnx`: real inference from `models/onnx/model.onnx`
- `mock`: explicit fallback mode when no trained/exported model exists yet

No real accuracy/F1 claims are made until training and evaluation are run locally on the downloaded dataset.

## Architecture

```text
Kaggle PlantVillage
        |
        v
data/raw/PlantVillage -> src/data/split_dataset.py -> data/processed/train|val|test
        |
        v
src/training/train.py -> models/checkpoints/best_model.pth
        |
        v
src/inference/predict.py --export -> models/onnx/model.onnx
        |
        v
FastAPI backend -> SQLite disease info + scan history -> React frontend
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

## Data Download

Set Kaggle credentials:

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
python -m src.data.download_data
```

If Kaggle CLI is unavailable, download the PlantVillage zip manually, place it in `data/raw/`, then run:

```powershell
python -m src.data.download_data --skip-download
```

Split the dataset:

```powershell
python -m src.data.split_dataset
```

## Train

```powershell
python -m src.training.train --config configs/base.yaml
```

Training writes:

- `models/checkpoints/best_model.pth`
- `docs/training_logs/training_log.csv`
- `docs/training_logs/training_log.json`
- `docs/training_curves.png`

## Evaluate

```powershell
python -m src.evaluation.evaluate --checkpoint models/checkpoints/best_model.pth
```

This writes `docs/model_performance_report.md` and `docs/confusion_matrix.png`.

## Export ONNX

```powershell
python -m src.inference.predict --checkpoint models/checkpoints/best_model.pth --output models/onnx/model.onnx
```

## Run Backend

```powershell
python backend/db/seed_disease_data.py
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

## Run Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

## Model Performance

| Run | Architecture | Test Accuracy | Macro F1 | Notes |
|---|---|---:|---:|---|
| pending | efficientnet_b0 | not run | not run | Run training/evaluation after downloading PlantVillage |

## Known Limitations

- PlantVillage images are often lab-condition leaf photos; field accuracy may be lower under variable lighting, backgrounds, occlusion, or mixed symptoms.
- Disease treatment text includes expert-review warnings where guidance is not verified.
- The app should support agronomist review before farmers make pesticide or crop-loss decisions.
- Until `models/onnx/model.onnx` exists, backend inference runs in explicit mock fallback mode.
