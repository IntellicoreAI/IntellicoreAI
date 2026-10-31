FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN pip install --upgrade pip && \
    pip install \
    fastapi \
    pydantic \
    uvicorn \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    scipy \
    shap \
    lime \
    supabase \
    python-dotenv

COPY . /workspace

WORKDIR /workspace/services/ml-monitoring

CMD ["python", "-m", "app.main"]
