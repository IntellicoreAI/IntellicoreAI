from uuid import uuid4
from .evaluators.groundedness import score_groundedness
from .evaluators.hallucination import evaluate_hallucination
from .evaluators.relevance import evaluate_relevance
from .evaluators.response_quality import evaluate_response_quality
from .evaluators.safety import evaluate_safety, redact_sensitive_text
from .metrics.latency import latency_metrics
from .metrics.tokens import token_metrics
from .metrics.cost import cost_metrics
from .metrics.errors import error_metrics
from .schemas.event import EvaluateRequest, LLMEvent
from .storage import save_event
from .compliance import fintech_checks

def evaluate_request(req: EvaluateRequest) -> dict:
    trace_id, span_id = req.trace_id or str(uuid4()), req.span_id or str(uuid4())
    latency = latency_metrics(req.latency_ms)
    tokens = token_metrics(req.input_tokens, req.output_tokens)
    cost = cost_metrics(req.cost_usd, req.input_tokens, req.output_tokens)
    errors = error_metrics(req.status, req.error)
    evaluations = {
      "groundedness": score_groundedness(req.response, req.context),
      "hallucination": evaluate_hallucination(req.response, req.context),
      "relevance": evaluate_relevance(req.prompt, req.response),
      "response_quality": evaluate_response_quality(req.response, req.expected_format, req.citations),
      "safety": evaluate_safety(req.prompt, req.response, req.metadata),
    }
    scores = [v["score"] for v in evaluations.values() if isinstance(v.get("score"), (float,int))]
    risk_flags=[]
    if evaluations["hallucination"]["label"] == "high_risk": risk_flags.append("hallucination_risk")
    if evaluations["safety"]["flags"]: risk_flags.extend(evaluations["safety"]["flags"])
    if errors["is_error"]: risk_flags.append("request_error")
    if latency["latency_band"] == "slow": risk_flags.append("high_latency")
    if evaluations["response_quality"].get("toxicity_rule_triggered"): risk_flags.append("possible_toxicity")
    if evaluations["response_quality"].get("bias_rule_triggered"): risk_flags.append("possible_bias")
    if evaluations["response_quality"].get("json_format_valid") is False: risk_flags.append("invalid_json_output")
    overall = round(sum(scores)/len(scores), 4) if scores else None
    fintech = fintech_checks(req.prompt, req.response)
    if fintech["flags"]: risk_flags.append("fintech_compliance_screen")
    result = {"event_id": str(uuid4()), "trace_id": trace_id, "span_id": span_id,
      "project_id": req.project_id, "model_id": req.model_id, "evaluations": evaluations,
      "metrics": {**latency, **tokens, **cost, **errors,
        "time_to_first_token_ms": req.time_to_first_token_ms,
        "throughput_tokens_per_second": req.throughput_tokens_per_second,
        "prompt_length_chars": len(req.prompt), "prompt_length_words": len(req.prompt.split()),
        "context_document_count": len(req.retrieved_documents) if req.retrieved_documents else (len(req.context) if isinstance(req.context, list) else int(bool(req.context))),
        "context_present": bool(req.context) or bool(req.retrieved_documents),
        "prompt_injection_risk": "flagged" if "possible_prompt_injection" in evaluations["safety"]["flags"] else "no_rule_triggered",
        "privacy_risk_flags": [f for f in evaluations["safety"]["flags"] if "data" in f or "credential" in f or "pii" in f or "financial" in f],
        "prompt_type": req.prompt_type, "language": req.language, "locale": req.locale,
        "environment": req.environment, "region": req.region},
      "risk_flags": sorted(set(risk_flags)), "overall_heuristic_score": overall, "fintech_compliance": fintech,
      "human_review_recommended": bool(risk_flags) or (overall is not None and overall < .6),
      "limitations": ["Scores are heuristic signals, not proof of factuality or safety.",
                      "Human review is required for high-impact financial decisions."]}
    metadata = dict(req.metadata)
    metadata.update({"evaluations": evaluations, "metrics": result["metrics"], "risk_flags": result["risk_flags"],
                     "human_review_recommended": result["human_review_recommended"], "fintech_compliance": fintech, "user_id": req.user_id,
                     "session_id": req.session_id, "temperature": req.temperature, "top_p": req.top_p,
                     "top_k": req.top_k, "max_tokens": req.max_tokens, "stop_sequences": req.stop_sequences,
                     "prompt_type": req.prompt_type, "language": req.language, "locale": req.locale,
                     "environment": req.environment, "region": req.region, "expected_format": req.expected_format,
                     "citations": req.citations, "retrieved_document_metadata": req.retrieved_documents})
    event = LLMEvent(event_id=result["event_id"], project_id=req.project_id, model_id=req.model_id,
      trace_id=trace_id, span_id=span_id, event_type="llm.evaluated",
      input={"prompt": redact_sensitive_text(req.prompt),
             "context": redact_sensitive_text(" ".join(req.context) if isinstance(req.context, list) else (req.context or "")),
             "retrieved_documents": [{**doc, "text": redact_sensitive_text(str(doc.get("text", "")))} for doc in req.retrieved_documents],
             "expected_answer": redact_sensitive_text(req.expected_answer or "")},
      output={"response": redact_sensitive_text(req.response), "evaluations": evaluations}, latency=req.latency_ms,
      status=req.status, error=req.error, metadata=metadata)
    save_event(event.model_dump(mode="json"))
    return result
