FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8014

CMD ["python", "app.py"]
