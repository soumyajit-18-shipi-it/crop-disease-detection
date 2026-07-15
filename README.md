# Leaflight — Crop Disease Detection

Leaflight is a full-stack crop disease detection platform for analyzing plant leaf images. It combines a React dashboard, a FastAPI inference API, SQLite-backed disease guidance and scan history, and a reproducible PyTorch training pipeline with ONNX export.

> [!IMPORTANT]
> The application and API run locally today, but real predictions require both `models/onnx/model.onnx` and `models/onnx/model.json`. If that bundle is missing, `/health` reports `model_loaded: false`, the model mode is `unavailable`, and prediction endpoints return HTTP 503. No production model or benchmark result is claimed until all candidate runs finish and pass ONNX parity checks.

## Latest Features

### Leaflight web app

- Drag-and-drop, file-picker, and rear-camera image capture for JPG, PNG, and WebP files up to 10 MB.
- Leaf preview with analysis/loading states and periodic backend availability checks.
- Top diagnosis, confidence score, low-confidence retake guidance, and expandable top-3 alternatives.
- Crop, severity, symptoms, and recommended-treatment guidance from SQLite.
- Helpful/not-helpful feedback logging.
- Recent scan field log plus a dashboard with total scans, most-common diagnosis, average confidence, healthy/diseased ratio, disease-frequency chart, and timestamped history cards.
- Responsive Scan, Dashboard, and About views.

### FastAPI backend

- Single-image and batch prediction endpoints backed by ONNX Runtime.
- Strict server-side image validation, configurable upload limit, and SHA-256 image hashes.
- Health, supported-class, disease-information, scan-history, and feedback endpoints.
- Automatic SQLite schema/data seeding at startup.
- CORS configuration and rotating request logs.
- Interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### Training and data pipeline

- Registered multi-source loading for required PlantVillage data plus optional PlantDoc and human-validated field-survey data.
- Deterministic, persisted train/validation/test manifests with content-hash grouping to reduce duplicate leakage.
- A hard training gate that accepts field-survey records only when a reviewer marks `eligible_for_training=true`.
- Field-survey ingestion from Excel/CSV/TSV, image validation, label normalization, duplicate reporting, a local human-review UI, and an append-only decision audit trail.
- Resumable EfficientNetV2-S, ConvNeXt-Tiny, and ConvNeXt-Base training with AdamW, cosine warmup, effective-number class weighting, label smoothing, MixUp/CutMix, gradient accumulation/clipping, mixed precision, EMA, and macro-F1 early stopping.
- Backbone-native timm preprocessing plus field-oriented Albumentations for illumination, color, geometry, blur, shadow/fog, JPEG degradation, and partial occlusion.
- Full evaluation, temperature scaling, ECE/reliability diagrams, atomic checkpoints, parity-checked ONNX export, CPU/GPU latency measurement, and reproducible model comparison.
- Production selection weighted by 40% validation macro F1, 20% calibration quality, 15% ONNX CPU speed, 15% ONNX size, and 10% peak GPU memory.

## Architecture

```text
PlantVillage ───────────────┐
PlantDoc (optional) ────────┼─> dataset registry ─> persisted 70/15/15 split manifest
Field survey (validated) ───┘                              │
                                                          v
                      EfficientNetV2-S / ConvNeXt-Tiny / ConvNeXt-Base
                                                          │
                                                          v
                         metrics + ONNX parity/CPU benchmark + model selection
                                                          │
                                                          v
React + Vite <── HTTP ──> FastAPI + ONNX Runtime <──> SQLite disease data/history
```

## Current Repository Status

- Persisted split: 20,638 PlantVillage images across 15 pepper, potato, and tomato classes.
- Split counts: 14,447 train, 3,097 validation, and 3,094 test images (seed 42).
- PlantDoc and field-survey sources are optional and were skipped in the current split manifest.
- Phase 2.5 uses the same persisted split and a separate `crop_disease_phase2_5` experiment directory. No accuracy/F1 value is reported until its on-disk candidate artifact is complete.
- See `docs/model_comparison.md`, `docs/training_results.md`, `docs/production_model.md`, and `docs/training_pipeline_audit.md` for measured status and engineering decisions without estimated metrics.

## Screenshots

### Scan workspace

![Leaflight scan workspace](docs/screenshots/leaflight-scan.png)

### Analytics dashboard

![Leaflight analytics dashboard](docs/screenshots/leaflight-dashboard.png)

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React 18, Vite 5, Axios, Recharts |
| API | FastAPI, Uvicorn, Pydantic, Pillow |
| Inference | ONNX Runtime, NumPy, OpenCV |
| Training | PyTorch, torchvision, timm, Albumentations, scikit-learn |
| Data/reporting | pandas, openpyxl, SQLite, Matplotlib |
| Testing | pytest, FastAPI TestClient |

## Quick Start

Run all commands from the repository root unless a section says otherwise.

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt

npm.cmd --prefix frontend install
```

Vite requires Node.js 18 or newer.

### 2. Configure the services

The frontend defaults to `http://127.0.0.1:8000`. To override it:

```powershell
$env:VITE_API_URL="http://127.0.0.1:8000"
```

Backend configuration is read from environment variables:

| Variable | Default |
|---|---|
| `MODEL_PATH` | `models/onnx/model.onnx` |
| `MODEL_METADATA_PATH` | sibling `model.json` file |
| `DB_PATH` | `backend/db/disease_info.db` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `MAX_UPLOAD_SIZE_MB` | `10` |

The included `.env.example` files are references; export the variables in your shell because the application does not auto-load `.env` files.

### 3. Prepare an ONNX bundle for predictions

If you have the legacy checkpoint at `models/checkpoints/best_model.pth`, export it with:

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict `
  --checkpoint models/checkpoints/best_model.pth `
  --output models/onnx/model.onnx
```

This writes both `model.onnx` and `model.json`. The newer production training workflow creates parity-verified bundles under `artifacts/training/`; point `MODEL_PATH` and `MODEL_METADATA_PATH` at the selected ONNX and metadata files when serving one of those bundles.

### 4. Run the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app `
  --reload --host 127.0.0.1 --port 8000
```

Database setup and disease-data seeding happen automatically at startup.

- Health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`

### 5. Run the frontend

In a second terminal:

```powershell
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API, model, and database status |
| `GET` | `/classes` | Classes loaded from model metadata |
| `POST` | `/predict` | Analyze one image |
| `POST` | `/predict/batch` | Analyze multiple images |
| `GET` | `/disease/{class_name}` | Disease details and guidance |
| `GET` | `/history?limit=50` | Recent scans (1–200 records) |
| `POST` | `/feedback` | Record result feedback |

## Dataset Workflow

### Download and split PlantVillage

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
python -m src.data.download_data
python -m src.data.split_dataset
```

If the Kaggle CLI is unavailable, place the downloaded archive in `data/raw/` and run:

```powershell
python -m src.data.download_data --skip-download
```

### Ingest and review field-survey data

```powershell
python -m src.data.ingest_field_survey `
  --survey-file path\to\survey.xlsx `
  --image-root path\to\survey-images

python -m src.data.clean_field_survey_labels
python -m src.data.review_field_survey --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`, review the grouped labels, and accept, replace, or reject them. Original records are retained; only accepted/replaced records become training-eligible.

## Train, Benchmark, and Export

Train or resume one configured architecture:

```powershell
python -m src.training.train `
  --config configs/training/phase2_5.yaml `
  --architecture efficientnetv2_s
```

Run or resume all missing candidates and regenerate comparison reports:

```powershell
python -m src.training.benchmark --config configs/training/phase2_5.yaml --train
```

Each run writes to `artifacts/training/crop_disease_phase2_5/<architecture>/`, including:

- `best.pt` and resumable `last.pt` checkpoints
- `history.json`, `history.csv`, and `training_history.csv`
- `metrics.json`, `classification_report.json`, and `classification_report.txt`
- `confusion_matrix.png`, `calibration.json`, and `reliability_diagram.png`
- `model.onnx` and `model.json` with ONNX parity/CPU benchmark metadata

The benchmark promotes a model only after all configured candidates finish on one split hash and pass the required checks. The production directory contains `best.pt`, `best.onnx`, `metadata.json`, `metrics.json`, `training_history.csv`, `confusion_matrix.png`, `classification_report.json`, and `calibration.json` plus a checksummed bundle manifest.

Swin-Tiny is supported as an optional candidate after the required three-model benchmark:

```powershell
python -m src.training.benchmark --config configs/training/phase2_5.yaml --train --include-optional
```

See `docs/future_model_roadmap.md` for the intentionally unimplemented dataset, segmentation, severity, explainability, recommendation, active-learning, and continual-learning roadmap.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
npm.cmd --prefix frontend run build
```

## Known Limitations

- PlantVillage contains many lab-condition leaf photos; field performance can be lower under variable lighting, clutter, occlusion, or mixed symptoms.
- Disease and treatment guidance is decision support, not a substitute for local agronomy advice.
- The current UI backend indicator reflects API availability; confirm `model_loaded` from `/health` before presenting the model as ready.
- Prediction endpoints intentionally fail closed when a complete ONNX bundle is unavailable.
- No model accuracy or F1 claim is made until the benchmark artifacts contain completed measurements.

See `docs/deployment.md` for deployment notes.
