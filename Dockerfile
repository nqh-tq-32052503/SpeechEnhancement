# ---- Base image: CUDA runtime for GPU inference ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# System deps (Python 3.11 + audio libs + git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Python packages (install GPU build of onnxruntime)
# Equivalent to:
# !pip install -q museval resampy
# !pip uninstall -q -y onnxruntime
# !pip install -q -U onnxruntime-gpu

RUN python3.11 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# -------- Python deps (mirror the notebook) --------
# Torch stack (CUDA 12.4 wheels)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

RUN python3 -m pip install \
        numpy \
        soundfile \
        resampy \
        museval \
        librosa

RUN pip install python-multipart

RUN pip install \
    fastapi==0.115.4 uvicorn[standard]==0.32.0 

# Optional: working directory for your code
WORKDIR /app
# If you have project files, uncomment the next line to copy them in:
COPY . /app

EXPOSE 8041
# Quick sanity check on container start (prints available ORT providers)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8041"]
