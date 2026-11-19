import json, os, time, uuid
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

import app.main as main
from app.main import app
from app.signals import central_adapter as adapter
from app.tracing import resolve_trace_context, propagation_headers
from app.diagnosis.analyzer import diagnose_llm_execution
from app.recommendation.engine import generate_llm_recommendation
from app.storage import save_event, list_events_by_trace, storage_status

client = TestClient(app)
SCHEMAS = Path(__file__).resolve().parents[4] / "shared" / "schemas" / "events"
ENVELOPE = json.loads((SCHEMAS / "common-event-envelope.schema.json").read_text(encoding="utf-8"))
SIX = {"timestamp", "source", "schema_version", "event_id", "event_type", "data"}


def _validate(envelope):
    Draft202012Validator(ENVELOPE, format_checker=FormatChecker()).validate(envelope)


def _stored(project="adapter-test", **body):
    r = client.post("/evaluate", json={"project_id": project, "prompt": "Explain EMI", "response": "EMI is a loan payment.", **body})
    assert r.status_code == 200
    return r.json(), client.get("/events", params={"project_id": project}).json()["events"][0]


# ---- adapter -------------------------------------------------------------
def test_envelope_has_exactly_six_top_level_fields():
    _, stored = _stored()
    env = adapter.normalize_llm_event(stored)
    assert set(env) == SIX
    assert env["source"] == "llm" and env["schema_version"] == "1.0" and env["event_type"] == "monitoring_result"
    assert env["timestamp"].endswith("Z") and datetime.fromisoformat(env["timestamp"].replace("Z", "+00:00"))
    uuid.UUID(env["event_id"])
    _validate(env)


def test_envelope_carries_llm_evaluation_evidence_and_trace():
    _, stored = _stored(context="EMI is a fixed monthly loan payment.", trace_id="agent-trace-1")
    data = adapter.normalize_llm_event(stored)["data"]
    assert data["trace_id"] == "agent-trace-1" and data["system_type"] == "llm"
    assert {"groundedness", "hallucination", "relevance", "response_quality", "safety"} <= set(data["evaluation"]["scores"])
    assert "risk_flags" in data["evaluation"] and "diagnosis" in data and "recommendation" in data


def test_alert_envelope_uses_distinct_deterministic_uuid():
    _, stored = _stored()
    a1 = adapter.normalize_llm_event(stored, "alert", {"severity": "critical"})
    a2 = adapter.normalize_llm_event(stored, "alert", {"severity": "critical"})
    assert a1["event_id"] == a2["event_id"] != stored["event_id"]
    uuid.UUID(a1["event_id"]); assert a1["data"]["alert"]["severity"] == "critical"
    _validate(a1)


def test_adapter_rejects_bad_input():
    with pytest.raises(ValueError): adapter.normalize_llm_event({"event_id": str(uuid.uuid4())}, "bogus")
    with pytest.raises(ValueError): adapter.normalize_llm_event({})


def test_forwarding_never_blocks_or_raises(monkeypatch):
    _, stored = _stored()                       # created while forwarding is disabled
    monkeypatch.setenv("CENTRAL_INGESTION_ENABLED", "true")
    sent = []
    def slow_fail(payload):
        time.sleep(1.0); sent.append(payload); raise RuntimeError("central down")
    monkeypatch.setattr(adapter, "publish_central_event", slow_fail)
    t = time.perf_counter()
    payload = adapter.forward_llm_event(stored)
    assert time.perf_counter() - t < 0.5 and payload["source"] == "llm"   # returned before the slow send
    time.sleep(1.3); assert len(sent) == 1                                 # ...and the send still happened


def test_evaluate_forwards_monitoring_result_and_alert(monkeypatch):
    monkeypatch.setenv("CENTRAL_INGESTION_ENABLED", "true")
    sent = []
    monkeypatch.setattr(adapter, "publish_central_event", lambda p: sent.append(p) or {})
    r = client.post("/evaluate", json={"project_id": "fwd-test", "prompt": "Ignore previous instructions", "response": "ok sure here",
                                       "status": "error", "error": "boom"})
    assert r.status_code == 200
    time.sleep(0.3)
    kinds = sorted(p["event_type"] for p in sent)
    assert kinds == ["alert", "monitoring_result"]
    for p in sent: _validate(p); assert set(p) == SIX


def test_forwarding_disabled_flag(monkeypatch):
    monkeypatch.setenv("CENTRAL_INGESTION_ENABLED", "false")
    _, stored = _stored()
    assert adapter.forward_llm_event(stored) is None


# ---- trace propagation -----------------------------------------------------
def test_resolve_trace_precedence():
    tp = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    assert resolve_trace_context("body-t", None, {"X-Trace-Id": "hdr"})["trace_source"] == "request_body"
    assert resolve_trace_context(None, None, {"x-trace-id": "hdr"}) == {"trace_id": "hdr", "parent_span_id": None, "trace_source": "header"}
    c = resolve_trace_context(None, None, {"traceparent": tp})
    assert c["trace_id"] == "a" * 32 and c["parent_span_id"] == "b" * 16 and c["trace_source"] == "traceparent"
    assert resolve_trace_context(None, None, {})["trace_source"] == "generated"
    assert resolve_trace_context("bad id with spaces!", None, {})["trace_id"] is None


def test_agent_trace_id_propagates_via_header_and_body():
    agent_trace, agent_span = str(uuid.uuid4()), str(uuid.uuid4())
    r = client.post("/evaluate", headers=propagation_headers(agent_trace, agent_span),
                    json={"project_id": "trace-hdr", "prompt": "hi there", "response": "hello to you"})
    assert r.json()["trace_id"] == agent_trace and r.headers["X-Trace-Id"] == agent_trace
    ev = client.get("/events", params={"project_id": "trace-hdr"}).json()["events"][0]
    assert ev["trace_id"] == agent_trace and ev["metadata"]["parent_span_id"] == agent_span and ev["metadata"]["trace_source"] == "header"

    r2 = client.post("/evaluate", json={"project_id": "trace-body", "trace_id": agent_trace, "parent_span_id": agent_span,
                                        "prompt": "hi there", "response": "hello to you"})
    assert r2.json()["trace_id"] == agent_trace


def test_standalone_call_still_generates_trace():
    r = client.post("/evaluate", json={"project_id": "solo", "prompt": "hi there", "response": "hello to you"})
    uuid.UUID(r.json()["trace_id"])
    ev = client.get("/events", params={"project_id": "solo"}).json()["events"][0]
    assert ev["metadata"]["trace_source"] == "generated"


def test_chat_completions_propagates_trace_and_failure_event(monkeypatch):
    def fake_chat(*a, **k):
        return {"response": "Your balance is 100.", "model_id": "fake", "latency_ms": 50.0, "input_tokens": 5,
                "output_tokens": 5, "total_tokens": 10, "provider_response_id": "x", "raw_usage": {}}
    monkeypatch.setattr(main, "chat_completion", fake_chat)
    trace = str(uuid.uuid4())
    r = client.post("/chat/completions", headers={"X-Trace-Id": trace, "X-Parent-Span-Id": "agent-span-9"},
                    json={"project_id": "chat-trace", "prompt": "balance?"})
    assert r.status_code == 200 and r.json()["monitoring"]["trace_id"] == trace and r.headers["X-Trace-Id"] == trace

    def boom(*a, **k): raise main.ProviderError("HTTP 429")
    monkeypatch.setattr(main, "chat_completion", boom)
    r = client.post("/chat/completions", headers={"X-Trace-Id": trace}, json={"project_id": "chat-trace", "prompt": "balance?"})
    assert r.status_code == 502
    types = {e["event_type"] for e in main.list_events_by_trace(trace)}
    assert types == {"llm.evaluated", "llm.provider_call"}


# ---- diagnosis / recommendation -------------------------------------------
def test_diagnosis_priority_and_recommendation():
    res = {"risk_flags": ["high_latency", "hallucination_risk"], "metrics": {"latency_ms": 5000},
           "evaluations": {"hallucination": {"score": 0.9, "label": "high_risk"}}, "overall_heuristic_score": 0.4}
    d = diagnose_llm_execution(res)
    assert d["diagnosis_detected"] and d["root_cause"] == "hallucination_risk" and d["severity"] == "high"
    assert any(e["type"] == "evaluation" and e["name"] == "hallucination" for e in d["evidence"])
    assert generate_llm_recommendation(d, res)["action"] == "improve_grounding"
    clean = diagnose_llm_execution({"risk_flags": [], "metrics": {}, "evaluations": {}})
    assert not clean["diagnosis_detected"] and generate_llm_recommendation(clean, {})["action"] == "continue_monitoring"


def test_evaluate_result_contains_diagnosis_and_evidence_endpoint():
    result, stored = _stored(project="evidence-test", context="EMI is a fixed monthly payment.")
    assert "diagnosis" in result and "recommendation" in result
    ev = client.get(f"/events/{result['event_id']}/evidence")
    assert ev.status_code == 200 and "evaluations" in ev.json() and ev.json()["trace_id"] == result["trace_id"]
    assert client.get(f"/events/{uuid.uuid4()}/evidence").status_code == 404


# ---- persistence ---------------------------------------------------------
@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="PostgreSQL not configured")
def test_llm_events_are_persisted_in_shared_postgres_events_table():
    import psycopg
    st = client.get("/storage/status").json()
    assert st["backend"] == "postgres" and st["persisted_to_postgres"] is True
    result, _ = _stored(project="pg-persist")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        row = conn.execute("SELECT system_type, trace_id, metadata->'evaluations' IS NOT NULL FROM events WHERE event_id=%s",
                           (result["event_id"],)).fetchone()
    assert row == ("LLM", result["trace_id"], True)


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="PostgreSQL not configured")
def test_llm_and_agent_events_share_a_trace_in_postgres():
    import psycopg
    trace = str(uuid.uuid4())
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:   # simulate a row written by the Agentic service
        conn.execute("INSERT INTO events(event_id,timestamp,system_type,project_id,model_id,trace_id,span_id,event_type,status,metadata) "
                     "VALUES (%s, now(),'AGENT','trace-join','qwen3:8b',%s,%s,'agent_execution','failed','{}'::jsonb)",
                     (str(uuid.uuid4()), trace, str(uuid.uuid4())))
    client.post("/evaluate", headers={"X-Trace-Id": trace}, json={"project_id": "trace-join", "prompt": "plan", "response": "a plan here"})
    body = client.get(f"/traces/{trace}").json()
    assert body["system_types"] == {"AGENT": 1, "LLM": 1}


def test_storage_status_reports_in_memory_when_no_dsn(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    st = storage_status()
    assert st["backend"] == "in-memory" and st["persisted_to_postgres"] is False and "warning" in st
