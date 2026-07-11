FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY data/class_mapping.json ./data/class_mapping.json
COPY models/model_config.json ./models/model_config.json
COPY models/onnx ./models/onnx

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
