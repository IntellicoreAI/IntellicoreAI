FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN pip install --upgrade pip && \
    pip install \
    fastapi \
    uvicorn \
    jsonschema \
    "supabase>=2.33.0" \
    python-dotenv

COPY . /workspace

WORKDIR /workspace/services/ingestion-service

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
