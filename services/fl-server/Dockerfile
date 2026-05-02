FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server_entrypoint.py .
COPY real_runner_wrapper.py .
COPY metrics.py .
COPY model_registry.py .
COPY node_registry.py .
COPY tier_assignment.py .

RUN useradd --uid 1000 --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080
EXPOSE 8000

CMD ["uvicorn", "server_entrypoint:app", "--host", "0.0.0.0", "--port", "8080"]
