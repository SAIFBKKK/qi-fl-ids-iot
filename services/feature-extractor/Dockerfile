# ── Stage 1 : builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2 : runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copier les dépendances compilées depuis le builder
COPY --from=builder /install /usr/local

# Copier uniquement les modules Python du service
COPY feature_map.py extractor.py ./

# Utilisateur non-root
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3).read()"

CMD ["python", "extractor.py"]
