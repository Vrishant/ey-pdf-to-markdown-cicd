# Location: Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

ARG BAKE_MODELS=false
COPY scripts/prefetch_models.py ./scripts/prefetch_models.py
RUN if [ "$BAKE_MODELS" = "true" ]; then python3 scripts/prefetch_models.py; fi

COPY . .

ENV PDF2MD_MODEL_DIR=/app/models

EXPOSE 8080

ENTRYPOINT ["python3", "-m", "pdf2md.serve"]
