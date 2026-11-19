"""Cross-paradigm rules: correlate LLM and Agentic findings that share a trace_id.

Correlation is only meaningful because the LLM monitoring service adopts the agent's
trace_id (body, X-Trace-Id header or W3C traceparent). Events without a shared trace_id
are never correlated here.
"""
from datetime import datetime, timezone
from typing import Any
from ..models import Finding
from . import agent_rules, llm_rules

LLM_QUALITY_CAUSES = {"hallucination_risk", "low_response_quality", "invalid_output_format"}
LLM_SAFETY_CAUSES = {"prompt_injection_detected", "sensitive_data_exposure", "harmful_content", "policy_violation"}
TOOL_CAUSES = {"tool_execution_failure", "tool_failure_detected_by_agent"}


def _ts(event: dict[str, Any]) -> datetime | None:
    v = event.get("timestamp")
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


def _llm_precedes_agent(llm_findings: list[Finding], agent_events: list[dict[str, Any]], events_by_id: dict[str, dict]) -> bool | None:
    llm_ts = [t for f in llm_findings for i in f.event_ids if (t := _ts(events_by_id.get(i, {})))]
    ag_ts = [t for e in agent_events if (t := _ts(e))]
    if not llm_ts or not ag_ts:
        return None
    return min(llm_ts) <= max(ag_ts)


def _cross(rule_id, root_cause, category, severity, conf, title, explanation, trace_id, llm_f, agent_f, action, ordered):
    evidence = [{"source": "cross", "type": "shared_trace", "trace_id": trace_id,
                 "llm_events": sorted({i for f in llm_f for i in f.event_ids}),
                 "agent_events": sorted({i for f in agent_f for i in f.event_ids}),
                 "llm_precedes_agent": ordered}]
    for f in llm_f + agent_f:
        evidence.extend(f.evidence)
    if ordered:
        conf = min(0.95, conf + 0.05)
    return Finding(rule_id=rule_id, paradigm="cross", root_cause=root_cause, category=category, severity=severity,
                   base_confidence=conf, title=title, explanation=explanation, trace_id=trace_id,
                   event_ids=sorted({i for f in llm_f + agent_f for i in f.event_ids}), evidence=evidence, recommended_action=action)


def evaluate_trace(trace_id: str | None, events: list[dict[str, Any]]) -> list[Finding]:
    """Findings that exist only because LLM and agent events share one trace."""
    if not trace_id:
        return []
    llm_events = [e for e in events if llm_rules.is_llm(e)]
    agent_events = [e for e in events if agent_rules.is_agent(e)]
    if not llm_events or not agent_events:
        return []
    by_id = {str(e.get("event_id")): e for e in events}
    llm_f = [f for e in llm_events for f in llm_rules.evaluate_event(e)]
    agent_f = [f for e in agent_events for f in agent_rules.evaluate_event(e)]
    ordered = _llm_precedes_agent(llm_f, agent_events, by_id)
    by_cause = lambda fs, causes: [f for f in fs if f.root_cause in causes]  # noqa: E731
    out: list[Finding] = []
    used: set[str] = set()

    provider = by_cause(llm_f, {"provider_request_failure"})
    if provider and agent_f:
        out.append(_cross("cross.llm_provider_failure_to_agent_failure", "llm_provider_failure_caused_agent_failure", "reliability", "critical", 0.85,
                          "LLM provider failure propagated to the agent run",
                          "An LLM provider request failed inside this agent trace and the agent execution then failed.",
                          trace_id, provider, agent_f, "retry_or_failover_provider", ordered)); used.add("provider")

    hall = by_cause(llm_f, {"hallucination_risk"})
    tool = by_cause(agent_f, TOOL_CAUSES)
    if hall and tool:
        out.append(_cross("cross.hallucination_to_tool_failure", "llm_hallucination_caused_tool_failure", "quality", "high", 0.75,
                          "Hallucinated LLM output likely drove a tool failure",
                          "The LLM response was weakly grounded and a tool call in the same agent trace then failed, "
                          "which is consistent with fabricated tool arguments.", trace_id, hall, tool, "improve_grounding", ordered)); used.add("hall")

    quality = by_cause(llm_f, LLM_QUALITY_CAUSES)
    loop = by_cause(agent_f, {"repeated_execution_loop"})
    if quality and loop:
        out.append(_cross("cross.low_quality_to_agent_loop", "llm_output_driving_agent_loop", "quality", "high", 0.7,
                          "Low-quality LLM output likely keeps the agent looping",
                          "The agent repeated nodes while the LLM responses in the same trace were low quality or weakly grounded.",
                          trace_id, quality, loop, "refine_prompt_or_model", ordered)); used.add("loop")

    slow = by_cause(llm_f, {"high_latency"})
    if slow and agent_f and "provider" not in used:
        out.append(_cross("cross.llm_latency_to_agent_failure", "llm_latency_caused_agent_failure", "performance", "high", 0.6,
                          "Slow LLM calls coincide with an agent failure",
                          "LLM latency was high in a trace whose agent execution failed.", trace_id, slow, agent_f, "reduce_latency", ordered)); used.add("slow")

    safety = by_cause(llm_f, LLM_SAFETY_CAUSES)
    if safety:  # unsafe content matters even when the agent itself succeeded
        out.append(_cross("cross.unsafe_llm_content_in_agent_run", "unsafe_llm_content_in_agent_run", "safety", "high", 0.7,
                          "Unsafe LLM content appeared inside an agent run",
                          "Prompt-injection, sensitive-data or harmful content was detected on an LLM call made by an agent; "
                          "the agent may have acted on it.", trace_id, safety, agent_events and agent_f or [], "block_and_review_input", ordered))

    if llm_f and agent_f and not out:  # weaker, correlation-only fallback
        out.append(_cross("cross.llm_issue_coincides_with_agent_failure", "llm_issue_coincides_with_agent_failure", "correlation", "medium", 0.5,
                          "LLM issue coincides with an agent failure",
                          "Both an LLM issue and an agent failure occurred in the same trace; causality is not established.",
                          trace_id, llm_f, agent_f, "investigate_trace", ordered))
    return out
