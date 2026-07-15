# Deployment

## Backend on Railway or Render

1. Build from the repository root using `backend/Dockerfile`.
2. Set environment variables:
   - `MODEL_PATH=models/onnx/model.onnx`
   - `MODEL_METADATA_PATH=models/onnx/model.json`
   - `DB_PATH=backend/db/disease_info.db`
   - `CORS_ORIGINS=https://your-frontend.vercel.app`
   - `MAX_UPLOAD_SIZE_MB=10`
3. Ensure `models/onnx/model.onnx`, `models/onnx/model.json`, and `data/class_mapping.json` are included in the deployed artifact.
4. Start command if not using Docker CMD:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

## Frontend on Vercel

1. Set project root to `frontend`.
2. Set environment variable:
   - `VITE_API_URL=https://your-backend-url`
3. Build command:

```bash
npm run build
```

4. Output directory:

```bash
dist
```

## Swapping in a Retrained Model

1. Complete the Phase 2.5 benchmark and production selection:

```powershell
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train
```

2. Deploy `artifacts/training/crop_disease_phase2_5/production/best.onnx` and `metadata.json` together. Set `MODEL_PATH` and `MODEL_METADATA_PATH` to those deployed paths.
3. Verify `/health` reports `model_loaded: true`, then run a known-image smoke test. The metadata carries backbone-specific preprocessing and the fitted calibration temperature; do not pair the ONNX file with metadata from another run.
4. The frontend does not need redeployment if the API contract remains unchanged.
