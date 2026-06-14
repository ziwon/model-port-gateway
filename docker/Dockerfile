FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY model_port ./model_port
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --upgrade pip

FROM base AS api

RUN pip install -e '.[api]'

CMD ["uvicorn", "model_port.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

FROM base AS trainer

RUN pip install -e '.[train,api]'

CMD ["sleep", "infinity"]

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS trainer-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip

COPY pyproject.toml README.md ./
COPY model_port ./model_port
COPY configs ./configs
COPY scripts ./scripts

RUN pip install -e '.[train,api]' \
    && pip uninstall -y onnxruntime \
    && pip install 'onnxruntime-gpu>=1.18'

CMD ["sleep", "infinity"]
