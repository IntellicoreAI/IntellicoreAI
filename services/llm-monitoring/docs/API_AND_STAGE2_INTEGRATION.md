# LLM API and Stage 2 event integration

## Shared event contract

LLM monitoring persists events against `shared/schemas/events/base-event.schema.json` without modifying that shared schema. Every stored LLM event uses `system_type: "LLM"` and includes the shared required fields: `event_id`, `timestamp`, `project_id`, `trace_id`, `span_id`, `event_type`, and `status`. The existing `events` table is reused.

## Example event

```json
{
  "event_id": "4d3d6a14-27c1-4a51-8cc1-5ecf0e5c64ad",
  "timestamp": "2026-10-30T12:00:00Z",
  "system_type": "LLM",
  "project_id": "fintech-demo",
  "model_id": "gpt-4o-mini",
  "trace_id": "trace-7d20",
  "span_id": "span-llm-01",
  "event_type": "llm.evaluated",
  "input": {"prompt": "Explain the EMI for a loan."},
  "output": {"response": "EMI is a fixed periodic loan payment."},
  "latency": 840.5,
  "status": "success",
  "error": null,
  "metadata": {
    "provider": "openai-compatible",
    "evaluations": {"groundedness": {"label": "not_evaluable"}},
    "metrics": {"input_tokens": 12, "output_tokens": 9, "total_tokens": 21},
    "risk_flags": []
  }
}
```

The example is illustrative; actual IDs, timestamps, usage, and evaluation values are generated at runtime.

## Demonstration: real provider request and persisted event

1. Configure `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, and `DATABASE_URL` in the service environment.
2. Start PostgreSQL and the LLM service using the setup guide.
3. In Swagger, execute `POST /chat/completions` with:

```json
{
  "project_id": "fintech-demo",
  "prompt": "Explain what EMI means in a simple sentence.",
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "max_tokens": 120
}
```

4. Copy `monitoring.event_id` from the response.
5. Execute `GET /events?project_id=fintech-demo` and confirm that the event is present with `system_type` set to `LLM`.
6. Execute `GET /summary?project_id=fintech-demo` and `GET /metrics` to inspect aggregated information.

## Event payload documentation

- `POST /events`: accepts the common event schema; LLM events must use `system_type: "LLM"`.
- `GET /events`: filters by optional `project_id` and `limit`.
- `POST /chat/completions`: makes a provider request and creates a telemetry event.
- `POST /evaluate`: evaluates an already-generated prompt/response pair and creates a telemetry event.
- `POST /feedback`: saves feedback linked to an existing LLM event.

Automated tests validate the shared base event schema, provider request flow (mocked provider), feedback persistence behavior, drift checks, compliance screening, and alert retrieval. A live provider demonstration still requires a valid provider API key and network access.
