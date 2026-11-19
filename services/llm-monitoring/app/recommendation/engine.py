from typing import Any

_ACTIONS = {
    "provider_request_failure": ("high", "retry_or_failover_provider",
        "The provider call failed. Retry with backoff, check credentials/quota, or fail over to another provider."),
    "prompt_injection_detected": ("high", "block_and_review_input",
        "Prompt-injection patterns were detected. Sanitize the input and review the request before acting on the output."),
    "sensitive_data_exposure": ("high", "redact_and_review_data",
        "Sensitive data was detected. Remove it from prompts/responses and review data-handling controls."),
    "policy_violation": ("high", "review_policy_violation",
        "A policy violation was flagged. Route the interaction for human review."),
    "harmful_content": ("high", "filter_and_review_output",
        "Harmful or biased content was detected. Suppress the response and review safety settings."),
    "hallucination_risk": ("high", "improve_grounding",
        "The answer is weakly grounded. Provide better retrieval context, constrain the prompt to the context, or require citations."),
    "fintech_compliance_flag": ("high", "compliance_review",
        "FinTech screening flagged the content. Require compliance review before use."),
    "invalid_output_format": ("medium", "enforce_output_schema",
        "Output format was invalid. Use structured-output mode or add schema validation with retry."),
    "low_response_quality": ("medium", "refine_prompt_or_model",
        "Response quality was low. Refine the prompt, adjust temperature, or try a stronger model."),
    "high_latency": ("medium", "reduce_latency",
        "Latency was high. Reduce max_tokens/context size, use a faster model, or enable streaming/caching."),
}


def generate_llm_recommendation(diagnosis: dict[str, Any], evaluation_result: dict[str, Any]) -> dict[str, Any]:
    """Actionable recommendation from an LLM diagnosis (mirrors generate_agent_recommendation)."""
    root_cause = diagnosis.get("root_cause")
    rec: dict[str, Any] = {"recommendation_detected": False, "priority": "low",
                           "action": "continue_monitoring",
                           "reason": "No actionable LLM issue was detected. Continue monitoring.",
                           "evidence": []}
    if root_cause in _ACTIONS:
        priority, action, reason = _ACTIONS[root_cause]
        rec.update({"recommendation_detected": True, "priority": priority, "action": action, "reason": reason})
    rec["evidence"] = [{"root_cause": root_cause, "severity": diagnosis.get("severity", "low"),
                        "risk_flags": evaluation_result.get("risk_flags", [])}]
    return rec
