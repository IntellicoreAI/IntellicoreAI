# IntelliCore LLM Monitoring Service

This service provides LLM telemetry ingestion, heuristic evaluation, OpenAI-compatible chat completion integration, durable feedback, statistical drift screening, embedding-based semantic similarity, FinTech rule screening, alerts, and Prometheus metrics.

## Run locally (from repository root)

```bash
docker compose up -d postgres
cd services/llm-monitoring
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# macOS/Linux instead: source .venv/bin/activate
pip install -e ".[dev]"
# Set DATABASE_URL and LLM_API_KEY in your shell or load them from a local .env file.
export DATABASE_URL="postgresql://intellicore:intellicore_dev@localhost:5433/intellicore"
export LLM_API_KEY="your-provider-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
python -m uvicorn app.main:app --reload --port 8001
```

Swagger: http://127.0.0.1:8001/docs

## Main endpoints

- `POST /chat/completions`: sends a real prompt to an OpenAI-compatible provider, evaluates the response, and persists the LLM event.
- `POST /evaluate`: evaluates a supplied prompt/response pair without making a provider call.
- `POST /events`, `GET /events`: event ingestion and retrieval.
- `POST /feedback`, `GET /feedback`: durable feedback (PostgreSQL); in-memory fallback only when DATABASE_URL is absent.
- `POST /drift/check`: compares numeric reference/current windows using a standardized mean-shift score.
- `POST /evaluate/semantic`: compares two texts using provider embeddings.
- `GET /alerts`: retrieves generated alerts.
- `GET /summary`, `GET /metrics`, `GET /health`.

## Provider configuration

The default provider is OpenAI Chat Completions. `LLM_BASE_URL` can point to another OpenAI-compatible provider. `LLM_API_KEY` (or `OPENAI_API_KEY`) is required for `/chat/completions` and `/evaluate/semantic`. Set token prices using `LLM_INPUT_PRICE_PER_1M_USD` and `LLM_OUTPUT_PRICE_PER_1M_USD`; cost is an estimate based on these configured rates and provider-reported token counts. Configure alert thresholds with `LLM_ALERT_LATENCY_MS` and `LLM_ALERT_QUALITY_MIN`.

## Shared event contract

Persisted LLM events use the existing `shared/schemas/events/base-event.schema.json` contract with `system_type="LLM"`. Agentic AI files and schemas are not modified by this LLM update.

## Important limitations

- Quality, safety, and FinTech checks are rule-based screening signals, not factuality proofs or regulatory certification.
- Drift endpoint expects explicit numeric reference and current windows. It does not silently infer a baseline.
- Semantic similarity uses provider embeddings and does not establish factual correctness.
- Alerts are persisted records; email/Slack/webhook delivery is not configured in this implementation.
- Use approved provider settings and avoid sending secrets, payment credentials, or unnecessary personal data to an external LLM.

See `docs/API_AND_STAGE2_INTEGRATION.md` for the shared Stage 2 event contract, example event, and end-to-end demonstration procedure.
