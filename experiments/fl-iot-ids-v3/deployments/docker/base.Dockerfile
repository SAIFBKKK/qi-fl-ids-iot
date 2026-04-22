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
COPY requirements-lock.txt requirements-lock.txt
COPY pyproject.toml pyproject.toml

RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch explicitly
RUN pip install --default-timeout=300 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies except torch
RUN grep -v '^torch' requirements.txt > /tmp/requirements-docker.txt && \
    pip install --default-timeout=300 -r /tmp/requirements-docker.txt

# Copy the whole project
COPY . .

# Install the local project
RUN pip install -e .

# Fail early if the restored explicit launchers are not importable
RUN python -c "import importlib; importlib.import_module('src.scripts.run_server'); importlib.import_module('src.scripts.run_client'); print('Explicit launchers import OK')"

WORKDIR /app