import math, os
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
load_dotenv()
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from .schemas.event import EvaluateRequest, LLMEvent, FeedbackRequest, ChatRequest, DriftRequest, SemanticRequest
from .service import evaluate_request
from .storage import (save_event, list_events, save_feedback, list_feedback, event_exists,
                      init_storage, save_alert, list_alerts)
from .provider import chat_completion, create_embeddings, ProviderError
from .drift import compare_windows
from .compliance import fintech_checks

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_storage()
    yield

app = FastAPI(title='IntelliCore LLM Monitoring Service', version='0.2.0',
              description='LLM provider integration, telemetry, quality/safety evaluation, drift screening and operational alerts.', lifespan=lifespan)
REQUESTS = Counter('intellicore_llm_requests_total', 'LLM requests evaluated', ['status'])
LATENCY = Histogram('intellicore_llm_latency_ms', 'LLM request latency in milliseconds', buckets=(50,100,250,500,1000,2000,3000,5000,10000))


def _estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    # Prices are USD per 1M tokens; configure these for the selected model.
    in_rate=float(os.getenv('LLM_INPUT_PRICE_PER_1M_USD','0'))
    out_rate=float(os.getenv('LLM_OUTPUT_PRICE_PER_1M_USD','0'))
    return round((input_tokens*in_rate + output_tokens*out_rate)/1_000_000, 10)


def _make_alerts(result: dict, fintech: dict | None = None):
    flags = list(result.get('risk_flags', []))
    metrics = result.get('metrics', {})
    latency_limit=float(os.getenv('LLM_ALERT_LATENCY_MS','3000'))
    quality_min=float(os.getenv('LLM_ALERT_QUALITY_MIN','0.60'))
    if metrics.get('latency_ms') is not None and metrics['latency_ms'] >= latency_limit: flags.append('high_latency')
    if result.get('overall_heuristic_score') is not None and result['overall_heuristic_score'] < quality_min: flags.append('low_response_quality')
    if fintech and fintech.get('flags'): flags.append('fintech_compliance_screen')
    for flag in sorted(set(flags)):
        severity = 'critical' if flag in {'request_error','possible_payment_card','possible_indian_aadhaar','possible_indian_pan','possible_bank_account'} else 'warning'
        save_alert({'event_id':result['event_id'],'project_id':result['project_id'],'severity':severity,
                    'alert_type':flag,'message':f'LLM monitoring alert: {flag.replace("_", " ")}',
                    'details':{'risk_flags':flags,'fintech_checks':fintech or {}}})

@app.get('/')
def home():
    return {'service':'IntelliCore LLM Monitoring Service','status':'running','docs':'/docs'}

@app.get('/health')
def health():
    return {'status':'healthy','service':'llm-monitoring','storage':'postgres' if os.getenv('DATABASE_URL') else 'in-memory-development',
            'provider_configured':bool(os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY'))}

@app.post('/events')
def ingest_event(event: LLMEvent):
    save_event(event.model_dump(mode='json'))
    return {'status':'accepted','event_id':event.event_id}

@app.get('/events')
def get_events(project_id: str | None = None, limit: int = Query(default=100, ge=1, le=1000)):
    rows=list_events(project_id,limit)
    return {'count':len(rows),'events':rows}

@app.post('/evaluate')
def evaluate(req: EvaluateRequest):
    result=evaluate_request(req)
    REQUESTS.labels(status='error' if result['metrics']['is_error'] else 'success').inc()
    if req.latency_ms is not None: LATENCY.observe(req.latency_ms)
    fintech=fintech_checks(req.prompt,req.response)
    result['fintech_compliance']=fintech
    _make_alerts(result,fintech)
    return result

@app.post('/chat/completions')
def monitored_chat(req: ChatRequest):
    """Call a real OpenAI-compatible provider, evaluate its answer and persist telemetry."""
    try:
        provider=chat_completion(req.prompt,model=req.model,system_prompt=req.system_prompt,temperature=req.temperature,max_tokens=req.max_tokens)
        evaluation=EvaluateRequest(project_id=req.project_id,model_id=provider['model_id'],trace_id=req.trace_id,
            prompt=req.prompt,response=provider['response'],context=req.context,latency_ms=provider['latency_ms'],
            input_tokens=provider['input_tokens'],output_tokens=provider['output_tokens'],
            cost_usd=_estimate_cost(provider['input_tokens'], provider['output_tokens']),
            status='success',user_id=req.user_id,temperature=req.temperature,max_tokens=req.max_tokens,
            metadata={**req.metadata,'provider':'openai-compatible','provider_response_id':provider['provider_response_id']})
        result=evaluate_request(evaluation)
        fintech=fintech_checks(req.prompt,provider['response'])
        result['fintech_compliance']=fintech
        _make_alerts(result,fintech)
        REQUESTS.labels(status='success').inc(); LATENCY.observe(provider['latency_ms'])
        return {'provider':{'model':provider['model_id'],'response_id':provider['provider_response_id']},
                'response':provider['response'],'usage':{'input_tokens':provider['input_tokens'],'output_tokens':provider['output_tokens'],'total_tokens':provider['total_tokens']},
                'monitoring':result}
    except ProviderError as exc:
        REQUESTS.labels(status='error').inc()
        # Persist a failed provider-call event too, so failures are observable.
        now=datetime.now(timezone.utc); event_id=str(uuid4())
        failed=LLMEvent(event_id=event_id,timestamp=now,project_id=req.project_id,model_id=req.model or os.getenv('LLM_MODEL'),
            trace_id=req.trace_id or str(uuid4()),span_id=str(uuid4()),event_type='llm.provider_call',
            input={'prompt':'[redacted from provider-error event]'},output=None,latency=None,status='error',error=str(exc),
            metadata={'provider':'openai-compatible','failure_stage':'provider_request','user_id':req.user_id})
        save_event(failed.model_dump(mode='json'))
        save_alert({'event_id':event_id,'project_id':req.project_id,'severity':'critical','alert_type':'provider_error',
                    'message':'LLM provider request failed','details':{'error':str(exc)}})
        raise HTTPException(status_code=502,detail={'message':str(exc),'event_id':event_id}) from exc

@app.get('/summary')
def summary(project_id: str | None = None, limit: int = Query(default=100,ge=1,le=1000)):
    rows=list_events(project_id,limit); total=len(rows)
    errors=sum(1 for row in rows if str(row.get('status','')).lower() not in {'success','ok','completed','200'} or row.get('error'))
    latencies=[float(r['latency']) for r in rows if r.get('latency') is not None]
    qualities=[]; hallucinations=[]; risk_counts={}
    for row in rows:
        md=row.get('metadata') or {}; ev=md.get('evaluations') or {}
        q=(ev.get('response_quality') or {}).get('score'); h=(ev.get('hallucination') or {}).get('score')
        if isinstance(q,(int,float)): qualities.append(q)
        if isinstance(h,(int,float)): hallucinations.append(h)
        for flag in md.get('risk_flags',[]): risk_counts[flag]=risk_counts.get(flag,0)+1
    return {'project_id':project_id,'sample_size':total,'success_rate':round((total-errors)/total,4) if total else None,
        'error_rate':round(errors/total,4) if total else None,'average_latency_ms':round(sum(latencies)/len(latencies),3) if latencies else None,
        'quality_score_average':round(sum(qualities)/len(qualities),4) if qualities else None,
        'hallucination_risk_average':round(sum(hallucinations)/len(hallucinations),4) if hallucinations else None,
        'risk_flag_counts':risk_counts}

@app.post('/feedback')
def feedback(req: FeedbackRequest):
    if not event_exists(req.event_id): raise HTTPException(status_code=404,detail='event_id not found')
    row=save_feedback(req.model_dump())
    return {'status':'recorded',**row}

@app.get('/feedback')
def get_feedback(event_id: str | None = None):
    rows=list_feedback(event_id)
    return {'count':len(rows),'feedback':rows}

@app.post('/drift/check')
def drift_check(req: DriftRequest):
    try: result=compare_windows(req.reference,req.current,req.threshold)
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    result.update({'project_id':req.project_id,'metric_name':req.metric_name})
    if result['drift_detected']:
        save_alert({'project_id':req.project_id,'severity':'warning','alert_type':'statistical_drift',
                    'message':f"Drift detected for {req.metric_name}",'details':result})
    return result

@app.post('/evaluate/semantic')
def semantic_evaluate(req: SemanticRequest):
    try:
        vectors=create_embeddings([req.reference_text,req.candidate_text],req.model)
    except ProviderError as exc: raise HTTPException(status_code=502,detail=str(exc)) from exc
    a,b=vectors
    denom=math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b))
    score=sum(x*y for x,y in zip(a,b))/denom if denom else 0.0
    return {'semantic_similarity':round(score,6),'label':'high_similarity' if score>=0.80 else 'moderate_similarity' if score>=0.60 else 'low_similarity',
            'embedding_model':req.model or os.getenv('LLM_EMBEDDING_MODEL','text-embedding-3-small'),
            'note':'Embedding similarity is a semantic signal, not a factuality or compliance guarantee.'}

@app.get('/alerts')
def get_alerts(project_id: str | None = None, limit: int = Query(default=100,ge=1,le=1000)):
    rows=list_alerts(project_id,limit)
    return {'count':len(rows),'alerts':rows}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)
