FROM python:3.12-slim
WORKDIR /app
COPY services/diagnosis-service/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic "psycopg[binary]" python-dotenv
COPY services/diagnosis-service/app ./app
ENV PYTHONPATH=/app
EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
