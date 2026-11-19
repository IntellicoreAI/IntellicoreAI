import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.engine.diagnosis_engine import DiagnosisEngine
from app.llm_judge import evaluator
from app.llm_judge.schemas import JudgeRequest
from app.main import app
from app.rules import agent_rules, cross_paradigm_rules, llm_rules

client = TestClient(app)


def llm_event(trace="t1", flags=(), status="success", latency=900, evals=None, ts="2026-11-19T10:00:00Z", **kw):
    return {"event_id": str(uuid.uuid4()), "timestamp": ts, "system_type": "LLM", "project_id": "p", "trace_id": trace,
            "event_type": "llm.evaluated", "status": status, "latency": latency, "input": "q", "output": "a",
            "metadata": {"risk_flags": list(flags), "evaluations": evals or {}, "metrics": {"cost_usd": 0.001, "total_tokens": 100}}, **kw}


def agent_event(trace="t1", status="failed", cause=None, ts="2026-11-19T10:00:05Z"):
    md = {"diagnosis": {"root_cause": cause, "explanation": "agent says so"}} if cause else {}
    return {"event_id": str(uuid.uuid4()), "timestamp": ts, "system_type": "AGENT", "project_id": "p", "trace_id": trace,
            "event_type": "agent_execution", "status": status, "latency": 3000, "metadata": md}


# ---- llm rules --------------------------------------------------------------
def test_llm_rules_flags_and_scores():
    causes = {f.root_cause for f in llm_rules.evaluate_event(llm_event(flags=["possible_prompt_injection", "possible_payment_card"]))}
    assert causes == {"prompt_injection_detected", "sensitive_data_exposure"}
    f = llm_rules.evaluate_event(llm_event(evals={"hallucination": {"score": 0.8, "label": "high_risk"}}))
    assert f[0].root_cause == "hallucination_risk" and any(e["type"] == "evaluation" for e in f[0].evidence)
    assert llm_rules.evaluate_event(llm_event()) == []
    assert llm_rules.evaluate_event(agent_event()) == []


def test_llm_window_rules():
    events = [llm_event(trace=f"x{i}", status="error" if i < 3 else "success") for i in range(8)]
    assert any(f.root_cause == "provider_instability" for f in llm_rules.evaluate_window(events))
    assert llm_rules.evaluate_window(events[:3]) == []     # below minimum sample


# ---- agent rules (read-only) -------------------------------------------------
def test_agent_rules_reuse_agent_diagnosis():
    f = agent_rules.evaluate_event(agent_event(cause="tool_execution_failure"))
    assert f[0].root_cause == "tool_execution_failure" and f[0].paradigm == "agent"
    assert agent_rules.evaluate_event(agent_event())[0].root_cause == "agent_execution_failure"
    assert agent_rules.evaluate_event(agent_event(status="success")) == []


# ---- cross-paradigm ------------------------------------------------------------
def test_cross_provider_failure_to_agent_failure():
    f = cross_paradigm_rules.evaluate_trace("t1", [llm_event(flags=["request_error"], status="error"), agent_event()])
    assert f[0].root_cause == "llm_provider_failure_caused_agent_failure" and f[0].severity == "critical"
    assert f[0].evidence[0]["type"] == "shared_trace" and f[0].evidence[0]["llm_precedes_agent"] is True


def test_cross_hallucination_to_tool_failure_and_loop():
    f = cross_paradigm_rules.evaluate_trace("t1", [llm_event(flags=["hallucination_risk"]), agent_event(cause="tool_execution_failure")])
    assert f[0].root_cause == "llm_hallucination_caused_tool_failure"
    f = cross_paradigm_rules.evaluate_trace("t1", [llm_event(flags=["low_response_quality"]), agent_event(cause="repeated_execution_loop")])
    assert f[0].root_cause == "llm_output_driving_agent_loop"


def test_cross_safety_even_when_agent_succeeds_and_no_cross_without_shared_events():
    f = cross_paradigm_rules.evaluate_trace("t1", [llm_event(flags=["possible_prompt_injection"]), agent_event(status="success")])
    assert f[0].root_cause == "unsafe_llm_content_in_agent_run"
    assert cross_paradigm_rules.evaluate_trace("t1", [llm_event(flags=["hallucination_risk"])]) == []
    assert cross_paradigm_rules.evaluate_trace(None, [llm_event(), agent_event()]) == []


def test_different_trace_ids_are_never_correlated():
    events = [llm_event(trace="A", flags=["request_error"], status="error"), agent_event(trace="B")]
    d = DiagnosisEngine().diagnose(events)
    assert all(not x.cross_paradigm for x in d)


# ---- engine ------------------------------------------------------------------
def test_engine_prefers_cross_paradigm_and_scores_confidence_and_impact():
    events = [llm_event(flags=["hallucination_risk"]), agent_event(cause="tool_execution_failure")]
    d = DiagnosisEngine().diagnose_trace("t1", events)
    assert d.cross_paradigm and d.root_cause == "llm_hallucination_caused_tool_failure"
    assert set(d.paradigms) == {"llm", "agent"} and 0.7 <= d.confidence <= 0.99
    assert d.impact["affected_event_count"] == 2 and d.impact["requires_human_review"] is True and d.impact["llm_tokens_involved"] == 100
    assert d.recommendation["action"] == "improve_grounding"
    assert {c["paradigm"] for c in d.contributing_findings} >= {"llm", "agent"}
    assert any(e.get("source") == "llm" and e["type"] == "evaluation" or e["type"] == "risk_flags" for e in d.evidence)


def test_engine_no_findings_returns_none_and_ids_are_deterministic():
    assert DiagnosisEngine().diagnose_trace("t1", [llm_event(), agent_event(status="success")]) is None
    ev = [llm_event(flags=["hallucination_risk"])]
    assert DiagnosisEngine().diagnose_trace("t1", ev).diagnosis_id == DiagnosisEngine().diagnose_trace("t1", ev).diagnosis_id


def test_standalone_llm_trace_diagnosed_as_llm_only():
    d = DiagnosisEngine().diagnose_trace("solo", [llm_event(trace="solo", flags=["fintech_compliance_screen"])])
    assert not d.cross_paradigm and d.paradigms == ["llm"] and d.root_cause == "fintech_compliance_flag"


# ---- judge ------------------------------------------------------------------
def _transport(text):
    return lambda url, headers, body, timeout: {"choices": [{"message": {"content": text}}]}


def test_judge_unavailable_without_key(monkeypatch):
    for k in ("JUDGE_LLM_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert evaluator.judge(JudgeRequest(prompt="p", response="r")).available is False


def test_judge_parses_verdict_and_adds_evidence():
    v = evaluator.judge(JudgeRequest(prompt="p", response="r"), _transport('Sure: {"issue_confirmed": true, "score": 0.9, "reasoning": "unsupported claim"}'))
    assert v.available and v.issue_confirmed and v.score == 0.9
    assert evaluator.judge(JudgeRequest(prompt="p", response="r"), _transport("not json")).available is False
    engine = DiagnosisEngine(judge=lambda evs: {"available": True, "issue_confirmed": True, "score": 0.9, "reasoning": "x"})
    ev = [llm_event(flags=["hallucination_risk"])]
    with_j = engine.diagnose_trace("t1", ev, use_judge=True)
    assert any(e["type"] == "llm_judge" for e in with_j.evidence)
    assert with_j.confidence > DiagnosisEngine().diagnose_trace("t1", ev).confidence
    no_j = DiagnosisEngine(judge=lambda evs: {"available": False}).diagnose_trace("t1", ev, use_judge=True)
    assert not any(e["type"] == "llm_judge" for e in no_j.evidence)


# ---- API --------------------------------------------------------------------
def test_api_diagnose_explicit_events_and_validation():
    r = client.post("/diagnose", json={"events": [llm_event(flags=["request_error"], status="error"), agent_event()]})
    assert r.status_code == 200 and r.json()["diagnoses"][0]["cross_paradigm"] is True
    assert client.post("/diagnose", json={}).status_code == 422
    assert client.get("/health").json()["service"] == "diagnosis-service"
    assert client.get("/diagnose/trace/none").status_code == 404
    assert client.get("/evidence/llm").status_code == 422


@pytest.mark.pg
@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="PostgreSQL not configured")
def test_end_to_end_postgres_llm_event_plus_agent_event():
    """Real LLM monitoring service -> shared events table -> diagnosis service."""
    import psycopg
    from app.repository import init_storage
    init_storage()
    import subprocess, sys, textwrap
    trace = str(uuid.uuid4())
    llm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../llm-monitoring"))
    # Both services are a package named `app`, so run the real LLM monitoring service in its own process.
    script = textwrap.dedent(f"""
        import os; os.environ["CENTRAL_INGESTION_ENABLED"] = "false"
        from fastapi.testclient import TestClient
        from app.main import app
        r = TestClient(app).post("/evaluate", headers={{"X-Trace-Id": "{trace}"}}, json={{
            "project_id": "e2e", "prompt": "What is the Aadhaar rule?", "context": "KYC needs ID.",
            "response": "Aadhaar was founded in 1850 by the Mughal emperor.", "status": "error", "error": "rate limited"}})
        assert r.status_code == 200 and r.json()["trace_id"] == "{trace}", r.text
    """)
    subprocess.run([sys.executable, "-c", script], cwd=llm_dir, check=True, env={**os.environ})
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("INSERT INTO events(event_id,timestamp,system_type,project_id,trace_id,span_id,event_type,status,metadata) "
                     "VALUES (%s, now() + interval '1 second','AGENT','e2e',%s,%s,'agent_execution','failed','{}'::jsonb)",
                     (str(uuid.uuid4()), trace, str(uuid.uuid4())))
    r = client.get(f"/diagnose/trace/{trace}")
    body = r.json()
    assert r.status_code == 200 and body["paradigms"] == ["agent", "llm"]
    assert body["diagnosis"]["cross_paradigm"] and body["diagnosis"]["root_cause"] == "llm_provider_failure_caused_agent_failure"
    assert client.get("/diagnoses", params={"trace_id": trace}).json()["count"] == 1
    client.get(f"/diagnose/trace/{trace}")                                   # re-run is idempotent
    assert client.get("/diagnoses", params={"trace_id": trace}).json()["count"] == 1
    ev = client.get("/evidence/llm", params={"trace_id": trace}).json()
    assert ev["count"] == 1 and any(e["type"] == "evaluation" for e in ev["evidence"][0]["evidence"])
