"""Diagnosis service: cross-paradigm (LLM + Agentic) diagnosis over the shared events table."""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from .engine.diagnosis_engine import DiagnosisEngine
from .engine.evidence import llm_evaluation_evidence
from .llm_judge.evaluator import judge, judge_events
from .llm_judge.schemas import JudgeRequest, JudgeVerdict
from .models import DiagnoseRequest
from .repository import dsn, fetch_events, init_storage, list_diagnoses, save_diagnosis
from .rules import llm_rules

load_dotenv()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_storage()
    yield


app = FastAPI(title="IntelliCore Diagnosis Service", version="0.1.0", lifespan=lifespan)
engine = DiagnosisEngine(judge=judge_events)


@app.get("/health")
def health():
    return {"status": "ok", "service": "diagnosis-service", "storage": "postgres" if dsn() else "in-memory"}


def _run(project_id, trace_id, events, use_judge, persist):
    diagnoses = engine.diagnose(events, project_id=project_id, use_judge=use_judge)
    if persist:
        for d in diagnoses:
            save_diagnosis(d.model_dump())
    return diagnoses


@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    """Diagnose explicit `events`, or a trace / project read from the shared events table."""
    if req.events is None and not (req.trace_id or req.project_id):
        raise HTTPException(status_code=422, detail="Provide events, trace_id or project_id.")
    events = req.events if req.events is not None else fetch_events(req.project_id, req.trace_id)
    diagnoses = _run(req.project_id, req.trace_id, events, req.use_judge, req.persist and req.events is None)
    return {"events_analyzed": len(events), "count": len(diagnoses), "diagnoses": [d.model_dump() for d in diagnoses]}


@app.get("/diagnose/trace/{trace_id}")
def diagnose_trace(trace_id: str, use_judge: bool = False, persist: bool = True):
    events = fetch_events(trace_id=trace_id)
    if not events:
        raise HTTPException(status_code=404, detail="No events found for this trace_id.")
    d = engine.diagnose_trace(trace_id, events, use_judge=use_judge)
    if d and persist:
        save_diagnosis(d.model_dump())
    return {"trace_id": trace_id, "events_analyzed": len(events),
            "paradigms": sorted({str(e.get("system_type", "")).lower() for e in events}),
            "diagnosis": d.model_dump() if d else None}


@app.get("/diagnose/project/{project_id}")
def diagnose_project(project_id: str, limit: int = Query(500, ge=1, le=5000), use_judge: bool = False, persist: bool = True):
    events = fetch_events(project_id=project_id, limit=limit)
    diagnoses = _run(project_id, None, events, use_judge, persist)
    return {"project_id": project_id, "events_analyzed": len(events), "count": len(diagnoses), "diagnoses": [d.model_dump() for d in diagnoses]}


@app.get("/diagnoses")
def get_diagnoses(project_id: str | None = None, trace_id: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    rows = list_diagnoses(project_id, trace_id, limit)
    return {"count": len(rows), "diagnoses": rows}


@app.get("/evidence/llm")
def llm_evidence(project_id: str | None = None, trace_id: str | None = None, limit: int = Query(200, ge=1, le=2000)):
    """LLM evaluation results as evidence packets for the Decision Pipeline."""
    if not (project_id or trace_id):
        raise HTTPException(status_code=422, detail="Provide project_id or trace_id.")
    events = [e for e in fetch_events(project_id, trace_id, limit) if llm_rules.is_llm(e)]
    packets = [{"event_id": str(e["event_id"]), "trace_id": e.get("trace_id"), "project_id": e.get("project_id"),
                "findings": [f.model_dump(include={"rule_id", "root_cause", "severity", "base_confidence"}) for f in llm_rules.evaluate_event(e)],
                "evidence": llm_evaluation_evidence(llm_rules.evaluation_evidence(e))} for e in events]
    return {"count": len(packets), "evidence": packets}


@app.post("/judge", response_model=JudgeVerdict)
def run_judge(req: JudgeRequest):
    """Optional LLM-as-judge verdict (extra evidence). Returns available=false when no judge key is configured."""
    return judge(req)
