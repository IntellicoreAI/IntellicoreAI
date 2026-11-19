"""LLM diagnosis rules. Consume LLM events (system_type='LLM') from the shared events table.

Evidence comes from the LLM monitoring service: metadata.evaluations, metadata.metrics,
metadata.risk_flags and metadata.fintech_compliance. Scores are heuristic signals.
"""
from typing import Any
from ..models import Finding

# risk_flag -> (rule_id, root_cause, category, severity, confidence, action)
FLAG_RULES = {
    "request_error": ("llm.provider_error", "provider_request_failure", "reliability", "high", 0.9, "retry_or_failover_provider"),
    "possible_prompt_injection": ("llm.prompt_injection", "prompt_injection_detected", "safety", "high", 0.75, "block_and_review_input"),
    "possible_payment_card": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.8, "redact_and_review_data"),
    "possible_indian_aadhaar": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.8, "redact_and_review_data"),
    "possible_indian_pan": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.8, "redact_and_review_data"),
    "possible_bank_account": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.75, "redact_and_review_data"),
    "possible_credential": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.8, "redact_and_review_data"),
    "possible_sensitive_data": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.7, "redact_and_review_data"),
    "pii_detected": ("llm.sensitive_data", "sensitive_data_exposure", "privacy", "high", 0.7, "redact_and_review_data"),
    "policy_violation": ("llm.policy_violation", "policy_violation", "safety", "high", 0.7, "review_policy_violation"),
    "possible_toxicity": ("llm.harmful_content", "harmful_content", "safety", "high", 0.7, "filter_and_review_output"),
    "possible_bias": ("llm.harmful_content", "harmful_content", "safety", "medium", 0.6, "filter_and_review_output"),
    "hallucination_risk": ("llm.hallucination", "hallucination_risk", "quality", "high", 0.7, "improve_grounding"),
    "fintech_compliance_screen": ("llm.fintech_compliance", "fintech_compliance_flag", "compliance", "high", 0.7, "compliance_review"),
    "invalid_json_output": ("llm.invalid_output", "invalid_output_format", "quality", "medium", 0.8, "enforce_output_schema"),
    "low_response_quality": ("llm.low_quality", "low_response_quality", "quality", "medium", 0.6, "refine_prompt_or_model"),
    "high_latency": ("llm.high_latency", "high_latency", "performance", "medium", 0.7, "reduce_latency"),
}

HALLUCINATION_SCORE = 0.65     # hallucination score at/above which risk is flagged without an explicit flag
QUALITY_SCORE_MIN = 0.45
LATENCY_MS = 5000
WINDOW_MIN_EVENTS = 5
WINDOW_ERROR_RATE = 0.3
WINDOW_HALLUCINATION_RATE = 0.3
WINDOW_AVG_LATENCY_MS = 4000


def is_llm(event: dict[str, Any]) -> bool:
    return str(event.get("system_type", "")).upper() == "LLM"


def _md(event: dict[str, Any]) -> dict[str, Any]:
    md = event.get("metadata")
    return md if isinstance(md, dict) else {}


def _eid(event: dict[str, Any]) -> str:
    return str(event.get("event_id"))


def evaluation_evidence(event: dict[str, Any]) -> list[dict[str, Any]]:
    """LLM evaluation results packaged as evidence for the Decision Pipeline."""
    md, out = _md(event), []
    for name, ev in (md.get("evaluations") or {}).items():
        if isinstance(ev, dict) and (ev.get("score") is not None or ev.get("label") not in (None, "not_evaluable")):
            out.append({"source": "llm", "type": "evaluation", "event_id": _eid(event), "name": name,
                        "score": ev.get("score"), "label": ev.get("label")})
    flags = md.get("risk_flags") or []
    if flags:
        out.append({"source": "llm", "type": "risk_flags", "event_id": _eid(event), "flags": sorted(set(flags))})
    latency = event.get("latency")
    if latency is not None:
        out.append({"source": "llm", "type": "latency", "event_id": _eid(event), "latency_ms": latency})
    if event.get("error"):
        out.append({"source": "llm", "type": "error", "event_id": _eid(event), "message": str(event["error"])[:300]})
    return out


def _finding(event, rule_id, root_cause, category, severity, conf, action, title, explanation) -> Finding:
    return Finding(rule_id=rule_id, paradigm="llm", root_cause=root_cause, category=category, severity=severity,
                   base_confidence=conf, title=title, explanation=explanation, trace_id=event.get("trace_id"),
                   event_ids=[_eid(event)], evidence=evaluation_evidence(event), recommended_action=action)


def evaluate_event(event: dict[str, Any]) -> list[Finding]:
    """All LLM findings for one event (a rule may fire once per root cause)."""
    if not is_llm(event):
        return []
    md = _md(event)
    flags = set(md.get("risk_flags") or [])
    evals = md.get("evaluations") or {}
    findings: dict[str, Finding] = {}

    if event.get("status") == "error" or event.get("event_type") == "llm.provider_call" or (md.get("metrics") or {}).get("is_error"):
        flags.add("request_error")
    for flag in sorted(flags):
        if flag in FLAG_RULES:
            rule_id, cause, cat, sev, conf, action = FLAG_RULES[flag]
            if cause not in findings:
                findings[cause] = _finding(event, rule_id, cause, cat, sev, conf, action,
                                           cause.replace("_", " ").capitalize(),
                                           f"LLM monitoring raised '{flag}' for this request.")

    # score-based rules fire even when the monitoring service did not set the flag
    hall = (evals.get("hallucination") or {}).get("score")
    if "hallucination_risk" not in flags and isinstance(hall, (int, float)) and hall >= HALLUCINATION_SCORE:
        findings["hallucination_risk"] = _finding(event, "llm.hallucination", "hallucination_risk", "quality", "high", 0.6,
                                                  "improve_grounding", "Hallucination risk",
                                                  f"Hallucination score {hall} is at or above {HALLUCINATION_SCORE}.")
    quality = (evals.get("response_quality") or {}).get("score")
    if "low_response_quality" not in flags and isinstance(quality, (int, float)) and quality < QUALITY_SCORE_MIN:
        findings["low_response_quality"] = _finding(event, "llm.low_quality", "low_response_quality", "quality", "medium", 0.55,
                                                    "refine_prompt_or_model", "Low response quality",
                                                    f"Response-quality score {quality} is below {QUALITY_SCORE_MIN}.")
    latency = event.get("latency")
    if "high_latency" not in flags and isinstance(latency, (int, float)) and latency >= LATENCY_MS:
        findings["high_latency"] = _finding(event, "llm.high_latency", "high_latency", "performance", "medium", 0.7,
                                            "reduce_latency", "High latency", f"Latency {latency} ms is at or above {LATENCY_MS} ms.")
    return list(findings.values())


def evaluate_window(events: list[dict[str, Any]]) -> list[Finding]:
    """Project-level LLM findings over a set of events (error/hallucination rate, average latency)."""
    llm = [e for e in events if is_llm(e)]
    if len(llm) < WINDOW_MIN_EVENTS:
        return []
    ids = [_eid(e) for e in llm][:50]
    out: list[Finding] = []
    errors = [e for e in llm if e.get("status") == "error"]
    if len(errors) / len(llm) >= WINDOW_ERROR_RATE:
        out.append(Finding(rule_id="llm.window.error_rate", paradigm="llm", root_cause="provider_instability", category="reliability",
                           severity="high", base_confidence=0.8, title="Elevated LLM error rate", event_ids=ids,
                           explanation=f"{len(errors)}/{len(llm)} LLM requests failed.",
                           evidence=[{"source": "llm", "type": "window_stat", "metric": "error_rate", "value": round(len(errors) / len(llm), 3), "n": len(llm)}],
                           recommended_action="retry_or_failover_provider"))
    hall = [e for e in llm if "hallucination_risk" in (_md(e).get("risk_flags") or [])]
    if len(hall) / len(llm) >= WINDOW_HALLUCINATION_RATE:
        out.append(Finding(rule_id="llm.window.hallucination_rate", paradigm="llm", root_cause="systemic_hallucination_risk", category="quality",
                           severity="high", base_confidence=0.75, title="Frequent hallucination risk", event_ids=ids,
                           explanation=f"{len(hall)}/{len(llm)} LLM responses were flagged for hallucination risk.",
                           evidence=[{"source": "llm", "type": "window_stat", "metric": "hallucination_rate", "value": round(len(hall) / len(llm), 3), "n": len(llm)}],
                           recommended_action="improve_grounding"))
    lat = [e["latency"] for e in llm if isinstance(e.get("latency"), (int, float))]
    if lat and sum(lat) / len(lat) >= WINDOW_AVG_LATENCY_MS:
        out.append(Finding(rule_id="llm.window.latency", paradigm="llm", root_cause="sustained_high_latency", category="performance",
                           severity="medium", base_confidence=0.7, title="Sustained high LLM latency", event_ids=ids,
                           explanation=f"Average LLM latency is {round(sum(lat) / len(lat))} ms.",
                           evidence=[{"source": "llm", "type": "window_stat", "metric": "avg_latency_ms", "value": round(sum(lat) / len(lat)), "n": len(lat)}],
                           recommended_action="reduce_latency"))
    return out
