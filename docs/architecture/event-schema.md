# Common event envelope

Every monitoring adapter (ml, llm, agentic) sends exactly six top-level fields to central ingestion
(schema: `shared/schemas/events/common-event-envelope.schema.json`).

| Field | Purpose | Example |
|---|---|---|
| timestamp | When the event occurred (UTC) | 2026-11-10T12:30:00Z |
| source | Module that generated it | ml, llm, agentic |
| schema_version | Contract version | 1.0 |
| event_id | Unique UUID | 6f1c... |
| event_type | monitoring_result or alert | monitoring_result |
| data | Module-specific JSON | evaluation scores, diagnosis, trace_id ... |

LLM adapter: `services/llm-monitoring/app/signals/central_adapter.py` (source = `llm`; alert ids are a deterministic
UUIDv5 of the monitoring event id). Forwarding is non-blocking and best-effort.
