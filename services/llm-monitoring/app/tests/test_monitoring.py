from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)

def test_health():
    r=client.get("/health")
    assert r.status_code == 200 and r.json()["service"] == "llm-monitoring"

def test_evaluate_all_fields():
    r=client.post("/evaluate", json={
      "project_id":"test-project", "model_id":"demo-model", "prompt":"What is the account balance?",
      "response":"The account balance is 120 dollars.", "context":"Account balance is 120 dollars.",
      "latency_ms":1250, "input_tokens":10, "output_tokens":8, "cost_usd":0.001,
      "metadata":{"environment":"test"}})
    assert r.status_code == 200
    data=r.json()
    assert set(data["evaluations"]) == {"groundedness","hallucination","relevance","response_quality","safety"}
    assert data["metrics"]["total_tokens"] == 18
    assert data["metrics"]["cost_usd"] == 0.001
    assert data["event_id"]

def test_no_context_not_evaluable():
    r=client.post("/evaluate", json={"prompt":"hello","response":"hello there"})
    assert r.status_code == 200
    assert r.json()["evaluations"]["groundedness"]["label"] == "not_evaluable"

def test_event_ingest():
    event={"project_id":"p","trace_id":"t","span_id":"s","event_type":"llm.request","status":"success"}
    r=client.post("/events",json=event)
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
