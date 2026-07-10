# USER MANUAL

# Crop Disease Detection

## Introduction

Crop Disease Detection is an AI-powered system that uses computer vision and deep learning models to detect crop diseases from plant images. The solution analyzes visual symptoms such as leaf spots, discoloration, and texture changes to accurately classify diseases.

---

# Installation & Getting Started

### Prerequisites
- Python 3.10+
- `uv` (recommended package manager) or `pip`

### Step 1: Install Dependencies
```bash
uv pip install -r requirements.txt
```
or
```bash
pip install -r requirements.txt
```

### Step 2: Running the FastAPI Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Accessing the API
Open your browser and navigate to:
- Swagger Docs: http://localhost:8000/docs
- Health Check endpoint: http://localhost:8000/

---

# Usage & Endpoints

### 1. Health Check
- **Endpoint**: `GET /`
- **Response**:
```json
{
  "status": "running",
  "service": "Crop Disease Detection API"
}
```

### 2. Predict Crop Disease
- **Endpoint**: `POST /predict`
- **Input**: A multipart form containing an image file.
- **Response**:
```json
{
  "filename": "tomato_leaf.jpg",
  "prediction": "Tomato Early Blight",
  "confidence": 0.94,
  "symptoms": ["leaf spots", "concentric rings", "yellowing margins"]
}
```

---

# Verification & Development Tasks

### Running Linting & Formatting
```bash
# Format check
uv run ruff format --check .
# Lint check
uv run ruff check .
```

### Running Static Type Check
```bash
uv run mypy .
```

### Running Security Audits
```bash
# SAST scan
uv run bandit -r app
```

### Running Unit Tests & Coverage
```bash
uv run pytest --cov=app --cov-report=term-missing --cov-fail-under=80
```
