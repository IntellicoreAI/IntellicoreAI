# IntelliCore AI – Unified Reliability & Intelligence Framework

IntelliCore AI is a unified framework designed to improve the reliability, monitoring and observability of production AI systems. It integrates three major components:

- **LLM Monitoring:** Evaluates LLM responses, tracks performance, detects risks and stores telemetry.
- **Agentic AI:** Handles agent-based workflows and intelligent task execution.
- **ML Monitoring:** Monitors machine learning models, performance and prediction behavior.

## 1. Project Status

| Component | Status |
|---|---|
| LLM Monitoring | Implemented and tested |
| Agentic AI | Under development |
| ML Monitoring | Under development |

### Completed LLM Features

- Groq API integration using GPT-OSS-20B.
- Response quality, relevance and hallucination-risk evaluation.
- Safety and FinTech compliance screening.
- Semantic evaluation using Sentence Transformers.
- Latency, token usage and API cost tracking.
- Drift detection and monitoring.
- Event generation and shared event-schema compatibility.
- PostgreSQL event persistence.
- Feedback collection and alert generation.
- Prometheus metrics.
- FastAPI and Swagger documentation.

## 2. Technology Stack

- **Language:** Python 3.11
- **Backend:** FastAPI
- **LLM Provider:** Groq API
- **Model:** GPT-OSS-20B
- **Database:** PostgreSQL
- **Semantic Evaluation:** Sentence Transformers
- **Metrics:** Prometheus
- **Testing:** Pytest
- **Containerization:** Docker and Docker Compose

## 3. Prerequisites

Install the following before running the project:

1. Python 3.11 or later
2. Git
3. Docker Desktop
4. Groq API key (required for real LLM requests)

## 4. Clone the Repository

Clone the required branch:

```bash
git clone -b <branch-name> <repository-url>
cd IntellicoreAI
```

Replace `<branch-name>` with the branch containing the latest project code.

## 5. Run the LLM Monitoring Service

### Step 1: Start PostgreSQL

Open a terminal in the project root directory:

```bash
docker compose up -d postgres
```

This starts the PostgreSQL database using the project's Docker Compose configuration.

### Step 2: Navigate to the LLM service

```bash
cd services/llm-monitoring
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows – Git Bash:**

```bash
source .venv/Scripts/activate
```

**Windows – Command Prompt:**

```cmd
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

Install the dependencies declared in the project's `pyproject.toml`:

```bash
pip install -e .
```

If the repository instead provides a `requirements.txt` as its installation source, use:

```bash
pip install -r requirements.txt
```

### Step 5: Configure environment variables

Create a `.env` file inside:

`services/llm-monitoring/`

Add:

```env
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_TIMEOUT_SECONDS=45

DATABASE_URL=postgresql://intellicore:intellicore_dev@localhost:5433/intellicore
```

Replace the API key with your own Groq API key. Never commit `.env` or expose API credentials on GitHub.

### Step 6: Start the FastAPI server

```bash
python -m uvicorn app.main:app --reload --port 8001
```

### Step 7: Access the API documentation

Open:

- Swagger UI: http://127.0.0.1:8001/docs
- OpenAPI JSON: http://127.0.0.1:8001/openapi.json
- Health Check: http://127.0.0.1:8001/health
- Prometheus Metrics: http://127.0.0.1:8001/metrics

## 6. Important LLM API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/chat/completions` | Generate LLM responses and telemetry |
| POST | `/evaluate` | Evaluate LLM responses |
| POST | `/events` | Ingest monitoring events |
| GET | `/events` | Retrieve stored events |
| GET | `/summary` | View monitoring summaries |
| GET | `/metrics` | View Prometheus metrics |
| POST | `/feedback` | Submit response feedback |
| GET | `/feedback` | Retrieve feedback |
| GET | `/alerts` | Retrieve monitoring alerts |
| POST | `/drift/check` | Detect metric drift |
| POST | `/evaluate/semantic` | Perform semantic evaluation |
| GET | `/health` | Check service health |

## 7. Run Automated Tests

From the `services/llm-monitoring` directory:

```bash
python -m pytest -v
```

The current LLM implementation has **13 passing automated tests**.

To check Python compilation:

```bash
python -m compileall -q app
```

## 8. Shared Event Schema

The common event schema is located at:

`shared/schemas/events/base-event.schema.json`

The LLM monitoring service uses this shared structure to maintain compatibility with the other project components.

**Important:** Do not modify the shared event schema independently. Coordinate any schema changes with the other team members.

## 9. Working on Agentic AI and ML Modules

Team members working on Agentic AI and ML should:

1. Clone the latest project branch.
2. Review the existing folder structure and shared event schema.
3. Work within their respective service directories.
4. Preserve compatibility with the shared event structure.
5. Avoid modifying another module's implementation without coordination.
6. Test their services independently before integrating them.

Each module should expose structured telemetry so that the overall framework can maintain a consistent monitoring architecture.

## 10. Important Notes

- Start Docker Desktop before running PostgreSQL.
- Start PostgreSQL before launching the LLM monitoring service.
- Configure your own Groq API key in `.env`.
- Do not upload `.env`, API keys or virtual environment folders to GitHub.
- The LLM monitoring service runs locally on port `8001`.
- Semantic similarity and compliance checks provide monitoring signals, not guaranteed factuality or regulatory certification.

---

**LLM Monitoring implementation: Completed and tested.**

IntelliCore AI aims to provide a unified framework for making AI systems more observable, measurable and reliable in production environments.
