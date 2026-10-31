

@'
# IntelliCore OS – ML Monitoring and Docker Setup

## 1. Project Overview

IntelliCore OS is a reliability and diagnosis layer designed to operate beneath client applications that use Machine Learning models.

The ML Monitoring component observes prediction telemetry, evaluates data quality, detects changes in feature and prediction distributions, monitors performance degradation, and generates monitoring results and alerts.

The current implementation uses FastAPI for ingestion, Supabase PostgreSQL for persistence, and Docker Compose for container orchestration.

## 2. System Architecture

The current ML Monitoring workflow:

1. Client ML application generates predictions.
2. Prediction events are created with input features, predictions, timestamps, and metadata.
3. FastAPI Ingestion Service receives and validates events.
4. Valid events are persisted in Supabase PostgreSQL.
5. ML Monitoring retrieves prediction telemetry from Supabase.
6. Monitoring detectors analyse the telemetry.
7. Monitoring results and alerts are sent through the Ingestion Service.
8. Results are persisted in Supabase.

### Architecture Flow

Client ML Application
        |
        v
Prediction Event Generator
        |
        v
FastAPI Ingestion Service
        |
        v
Supabase PostgreSQL
        |
        v
ML Monitoring Service
        |
        +---- Data Quality Monitoring
        |
        +---- Feature Drift Detection
        |
        +---- Prediction Drift Detection
        |
        +---- Performance Degradation Detection
        |
        v
Monitoring Results and Alerts
        |
        v
Ingestion Service
        |
        v
Supabase PostgreSQL

The current workflow is batch-oriented, not continuous.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Statistical Testing | SciPy |
| Explainability | SHAP, LIME |
| Validation | Pydantic, JSON Schema |
| Database | Supabase PostgreSQL |
| Database Client | Supabase Python Client |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Testing | Pytest |

## 4. Project Structure

```text
IntellicoreAI/
|
|-- docker-compose.yml
|-- .dockerignore
|-- .gitignore
|-- generate_dev_predictions.py
|-- sync_supabase_telemetry.py
|
|-- docs/
|   |-- ML_MONITORING_DOCKER_SETUP.md
|
|-- services/
|   |
|   |-- ingestion-service/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- app/
|   |   |   |-- main.py
|   |   |   |-- event_adapter.py
|   |   |   |-- monitoring_adapter.py
|   |   |   |-- supabase_client.py
|   |   |-- tests/
|   |
|   |-- ml-monitoring/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- app/
|   |   |   |-- main.py
|   |   |   |-- data_quality/
|   |   |   |-- monitoring/
|   |   |   |-- prediction/
|   |   |   |-- explainability/
|   |   |   |-- fairness/
|   |   |   |-- signals/
|   |   |   |-- telemetry/
|   |   |   |-- model_training/
|   |   |-- tests/
|   |
|   |-- diagnosis-service/
|       |-- app/
|       |   |-- diagnosis_engine.py
|       |-- tests/
|
|-- data/
    |-- artifacts/
    |-- processed/
    |-- telemetry/