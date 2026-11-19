from typing import Any

# Priority-ordered, mirroring the agentic diagnose_agent_execution() chain.
# (risk_flag, root_cause, category, severity, explanation)
_RULES = [
    ("request_error", "provider_request_failure", "reliability", "high",
     "The LLM request failed, so no usable model response was produced."),
    ("possible_prompt_injection", "prompt_injection_detected", "safety", "high",
     "The prompt or response matches prompt-injection patterns."),
    ("possible_payment_card", "sensitive_data_exposure", "privacy", "high",
     "A possible payment card number appeared in the prompt or response."),
    ("possible_indian_aadhaar", "sensitive_data_exposure", "privacy", "high",
     "A possible Aadhaar identifier appeared in the prompt or response."),
    ("possible_indian_pan", "sensitive_data_exposure", "privacy", "high",
     "A possible PAN identifier appeared in the prompt or response."),
    ("possible_bank_account", "sensitive_data_exposure", "privacy", "high",
     "A possible bank account number appeared in the prompt or response."),
    ("possible_credential", "sensitive_data_exposure", "privacy", "high",
     "A credential-like string appeared in the prompt or response."),
    ("possible_sensitive_data", "sensitive_data_exposure", "privacy", "high",
     "Sensitive numeric/identifier data appeared in the prompt or response."),
    ("possible_financial_data", "sensitive_data_exposure", "privacy", "high",
     "Financial account data appeared in the prompt or response."),
    ("pii_detected", "sensitive_data_exposure", "privacy", "high",
     "The caller flagged personally identifiable information."),
    ("sensitive_financial_data_detected", "sensitive_data_exposure", "privacy", "high",
     "The caller flagged sensitive financial data."),
    ("policy_violation", "policy_violation", "safety", "high",
     "The caller flagged a policy violation."),
    ("possible_toxicity", "harmful_content", "safety", "high",
     "The response matched toxicity rules."),
    ("possible_bias", "harmful_content", "safety", "medium",
     "The response matched bias rules."),
    ("hallucination_risk", "hallucination_risk", "quality", "high",
     "The response is weakly supported by the supplied context."),
    ("fintech_compliance_screen", "fintech_compliance_flag", "compliance", "high",
     "FinTech rule screening flagged the prompt or response; compliance review is required."),
    ("invalid_json_output", "invalid_output_format", "quality", "medium",
     "The response was expected to be valid JSON but could not be parsed."),
    ("low_response_quality", "low_response_quality", "quality", "medium",
     "The heuristic response-quality score is below the alert threshold."),
    ("high_latency", "high_latency", "performance", "medium",
     "The request latency exceeded the slow-latency band."),
]


def diagnose_llm_execution(evaluation_result: dict[str, Any]) -> dict[str, Any]:
    """Diagnose the primary reliability/quality/safety issue of one LLM request.

    Uses only observable evidence (metrics, heuristic evaluations, rule flags). It does
    not claim factual correctness; scores are heuristic signals.
    """
    flags = list(evaluation_result.get("risk_flags", []))
    evaluations = evaluation_result.get("evaluations", {})
    metrics = evaluation_result.get("metrics", {})
    overall = evaluation_result.get("overall_heuristic_score")

    diagnosis: dict[str, Any] = {
        "diagnosis_detected": False, "root_cause": "no_issue_detected", "category": "none",
        "severity": "low", "explanation": "No actionable LLM issue was detected from the available telemetry.",
        "evidence": [],
    }
    for flag, root_cause, category, severity, explanation in _RULES:
        if flag in flags:
            diagnosis.update({"diagnosis_detected": True, "root_cause": root_cause, "category": category,
                              "severity": severity, "explanation": explanation})
            break

    if diagnosis["diagnosis_detected"]:
        diagnosis["evidence"].append({"type": "risk_flags", "flags": sorted(set(flags))})
        for name in ("hallucination", "groundedness", "relevance", "response_quality", "safety"):
            ev = evaluations.get(name) or {}
            if ev.get("score") is not None or ev.get("label") not in (None, "not_evaluable"):
                diagnosis["evidence"].append({"type": "evaluation", "name": name,
                                              "score": ev.get("score"), "label": ev.get("label")})
        if metrics.get("is_error"):
            diagnosis["evidence"].append({"type": "error", "error_type": metrics.get("error_type"),
                                          "message": metrics.get("error_message")})
    diagnosis["evidence"].append({"type": "metrics", "latency_ms": metrics.get("latency_ms"),
                                  "total_tokens": metrics.get("total_tokens"),
                                  "overall_heuristic_score": overall})
    return diagnosis
