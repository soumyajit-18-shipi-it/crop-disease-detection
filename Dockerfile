# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install minimal system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY app /app/app

# Expose port 8000 for FastAPI
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
