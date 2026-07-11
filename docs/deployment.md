# Deployment

## Backend on Railway or Render

1. Build from the repository root using `backend/Dockerfile`.
2. Set environment variables:
   - `MODEL_PATH=models/onnx/model.onnx`
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

1. Train and evaluate a new checkpoint.
2. Export it to ONNX:

```bash
python -m src.inference.predict --checkpoint models/checkpoints/best_model.pth --output models/onnx/model.onnx
```

3. Deploy the backend artifact containing the new ONNX file and metadata JSON.
4. The frontend does not need redeployment if the API contract remains unchanged.
