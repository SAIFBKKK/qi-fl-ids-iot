FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch — avoids pulling a multi-GB CUDA image in CI / research
RUN pip install --default-timeout=300 \
    torch --index-url https://download.pytorch.org/whl/cpu

# All other dependencies (requirements.txt excludes torch)
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && \
    pip install --default-timeout=300 -r /tmp/req.txt

# Copy project source
COPY src/ src/
COPY configs/ configs/

# Install as editable package so `src.*` imports resolve
RUN pip install -e .

# Smoke-test imports at build time — fail fast on broken deps
RUN python -c "from src.scripts.run_experiment import main; print('imports OK')"
