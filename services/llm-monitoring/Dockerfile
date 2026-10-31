FROM python:3.12-slim
WORKDIR /app
COPY services/llm-monitoring/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir .
COPY services/llm-monitoring/app ./app
ENV PYTHONPATH=/app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
