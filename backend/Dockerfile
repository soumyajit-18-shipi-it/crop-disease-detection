FROM python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENVIRONMENT=production \
    LOG_TO_FILE=false

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && groupadd --system leaflight \
    && useradd --system --gid leaflight --home-dir /app --no-create-home leaflight

COPY --chown=leaflight:leaflight backend ./backend
COPY --chown=leaflight:leaflight src ./src
COPY --chown=leaflight:leaflight scripts/download_model.py ./scripts/download_model.py
COPY --chown=leaflight:leaflight data/class_mapping.json ./data/class_mapping.json
COPY --chown=leaflight:leaflight models/releases/efficientnetv2_s_v1 ./models/releases/efficientnetv2_s_v1

RUN python scripts/download_model.py \
    && python scripts/download_model.py --verify-only

EXPOSE 8000
USER leaflight

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=4).read()"

CMD ["sh", "-c", "exec python -m uvicorn backend.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
