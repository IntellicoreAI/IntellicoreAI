"""Agent rules for the shared diagnosis engine (read-only).

These do NOT modify or re-run the Agentic AI monitoring. They only interpret AGENT rows already
written to the shared events table, reusing the agent's own embedded diagnosis when present.
"""
from typing import Any
from ..models import Finding

AGENT_CAUSES = {
    "tool_execution_failure": ("tool", "high"),
    "repeated_execution_loop": ("loop", "high"),
    "agent_execution_failure": ("execution", "high"),
    "trace_execution_anomaly": ("trace", "medium"),
    "tool_failure_detected_by_agent": ("reasoning", "medium"),
}
BAD_STATUS = {"failed", "error", "failure", "timeout"}


def is_agent(event: dict[str, Any]) -> bool:
    return str(event.get("system_type", "")).upper() in {"AGENT", "AGENTIC", "AGENTIC_AI"}


def _md(event):
    md = event.get("metadata")
    return md if isinstance(md, dict) else {}


def evaluate_event(event: dict[str, Any]) -> list[Finding]:
    if not is_agent(event):
        return []
    md, eid = _md(event), str(event.get("event_id"))
    status = str(event.get("status") or "").lower()
    diag = md.get("diagnosis") if isinstance(md.get("diagnosis"), dict) else {}
    cause = diag.get("root_cause")
    evidence = [{"source": "agent", "type": "execution", "event_id": eid, "status": status or None, "latency_ms": event.get("latency")}]
    if event.get("error"):
        evidence.append({"source": "agent", "type": "error", "event_id": eid, "message": str(event["error"])[:300]})
    if diag.get("evidence"):
        evidence.append({"source": "agent", "type": "agent_diagnosis", "event_id": eid, "items": diag["evidence"]})

    if cause in AGENT_CAUSES:
        cat, sev = AGENT_CAUSES[cause]
        return [Finding(rule_id=f"agent.{cause}", paradigm="agent", root_cause=cause, category=cat, severity=sev, base_confidence=0.8,
                        title=cause.replace("_", " ").capitalize(), explanation=diag.get("explanation") or "Agent monitoring diagnosed this execution.",
                        trace_id=event.get("trace_id"), event_ids=[eid], evidence=evidence)]
    if status in BAD_STATUS:
        return [Finding(rule_id="agent.execution_failed", paradigm="agent", root_cause="agent_execution_failure", category="execution",
                        severity="high", base_confidence=0.7, title="Agent execution failed",
                        explanation=f"Agent execution ended with status '{status}'.", trace_id=event.get("trace_id"),
                        event_ids=[eid], evidence=evidence)]
    return []


def trace_failed(agent_events: list[dict[str, Any]]) -> bool:
    return any(evaluate_event(e) for e in agent_events)
