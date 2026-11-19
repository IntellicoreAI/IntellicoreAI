# Diagnosis engine

`services/diagnosis-service` reads events of all paradigms from the shared PostgreSQL `events` table.

    events -> rules (llm_rules, agent_rules, cross_paradigm_rules) -> findings
           -> root_cause ranking -> confidence -> impact -> Diagnosis (+ evidence, recommendation)

* Cross-paradigm rules only correlate events that share a `trace_id`. The LLM monitoring service adopts the agent's
  trace via request body, `X-Trace-Id` / `X-Parent-Span-Id` headers, or W3C `traceparent`.
* `agent_rules` is read-only: it interprets AGENT rows (reusing the agent's embedded diagnosis); agent monitoring is not modified.
* LLM evaluation results (groundedness, hallucination, relevance, quality, safety, risk flags, latency) are exposed as evidence
  in every diagnosis and at `GET /evidence/llm`.
* Optional LLM-as-judge adds evidence and nudges confidence; it is never required.
* `ml_rules.py` is intentionally left for the ML owner.

Endpoints (port 8003): `GET /health`, `POST /diagnose`, `GET /diagnose/trace/{id}`, `GET /diagnose/project/{id}`,
`GET /diagnoses`, `GET /evidence/llm`, `POST /judge`.
