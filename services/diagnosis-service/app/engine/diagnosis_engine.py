"""Cross-paradigm diagnosis engine: events -> findings -> ranked root cause -> Diagnosis."""
from collections import defaultdict
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ..models import Diagnosis, Finding
from ..rules import agent_rules, cross_paradigm_rules, llm_rules
from .confidence import score_confidence
from .evidence import merge_evidence
from .impact import assess_impact
from .root_cause import select_root_cause

ACTIONS = {
    "retry_or_failover_provider": "Retry with backoff, verify credentials/quota, or fail over to another LLM provider.",
    "block_and_review_input": "Sanitize the input and review the request before the agent acts on the output.",
    "redact_and_review_data": "Remove sensitive data from prompts/responses and review data-handling controls.",
    "review_policy_violation": "Route the interaction to human review.",
    "filter_and_review_output": "Suppress the response and review safety settings.",
    "improve_grounding": "Supply better retrieval context, constrain the prompt to it, and require citations; validate tool arguments before calling tools.",
    "compliance_review": "Require compliance review before the output is used.",
    "enforce_output_schema": "Use structured-output mode or schema validation with retry.",
    "refine_prompt_or_model": "Refine the prompt, lower temperature, or use a stronger model; add a loop guard to the agent.",
    "reduce_latency": "Reduce max_tokens/context, use a faster model, enable caching, and set agent timeouts.",
    "investigate_trace": "Inspect the shared trace to establish causality.",
}


class DiagnosisEngine:
    """Rule-based engine. `judge` (optional) is a callable(events)->dict used as extra LLM evidence."""

    def __init__(self, judge=None):
        self.judge = judge

    # -- findings -----------------------------------------------------------
    def collect_findings(self, events: list[dict[str, Any]]) -> list[Finding]:
        findings: list[Finding] = []
        for e in events:
            findings += llm_rules.evaluate_event(e) + agent_rules.evaluate_event(e)
        by_trace: dict[str | None, list[dict]] = defaultdict(list)
        for e in events:
            by_trace[e.get("trace_id")].append(e)
        for trace_id, group in by_trace.items():
            findings += cross_paradigm_rules.evaluate_trace(trace_id, group)
        return findings

    # -- one diagnosis from a set of findings ---------------------------------
    def _build(self, findings: list[Finding], events: list[dict[str, Any]], project_id, trace_id, use_judge=False) -> Diagnosis | None:
        primary, contributing = select_root_cause(findings)
        if primary is None:
            return None
        extra, judge_agrees = [], None
        if use_judge and self.judge:
            llm_events = [e for e in events if llm_rules.is_llm(e)]
            verdict = self.judge(llm_events) if llm_events else None
            if verdict and verdict.get("available"):
                extra.append({"source": "llm", "type": "llm_judge", **{k: v for k, v in verdict.items() if k != "available"}})
                judge_agrees = verdict.get("issue_confirmed")
        evidence = merge_evidence([primary] + contributing, extra)
        paradigms = sorted({p for f in [primary] + contributing for p in ({"llm", "agent"} if f.paradigm == "cross" else {f.paradigm})})
        action = primary.recommended_action
        return Diagnosis(
            diagnosis_id=str(uuid5(NAMESPACE_URL, f"intellicore-diagnosis:{project_id}:{trace_id}:{primary.root_cause}")), project_id=project_id, trace_id=trace_id, paradigms=paradigms,
            root_cause=primary.root_cause, category=primary.category, severity=primary.severity,
            confidence=score_confidence(primary, findings, judge_agrees),
            impact=assess_impact(primary, [primary] + contributing, events),
            explanation=primary.explanation or primary.title, evidence=evidence,
            contributing_findings=[{"rule_id": f.rule_id, "paradigm": f.paradigm, "root_cause": f.root_cause,
                                    "severity": f.severity, "event_ids": f.event_ids} for f in contributing],
            recommendation={"action": action, "detail": ACTIONS.get(action, ""), "priority": primary.severity} if action else {},
            cross_paradigm=primary.paradigm == "cross")

    # -- public API -----------------------------------------------------------
    def diagnose_trace(self, trace_id: str | None, events: list[dict[str, Any]], project_id=None, use_judge=False) -> Diagnosis | None:
        findings = self.collect_findings(events)
        return self._build(findings, events, project_id or next((e.get("project_id") for e in events), None), trace_id, use_judge)

    def diagnose(self, events: list[dict[str, Any]], project_id=None, use_judge=False) -> list[Diagnosis]:
        """One diagnosis per trace that has findings, plus one project-level diagnosis for window rules."""
        out: list[Diagnosis] = []
        by_trace: dict[str | None, list[dict]] = defaultdict(list)
        for e in events:
            by_trace[e.get("trace_id")].append(e)
        for trace_id, group in by_trace.items():
            d = self.diagnose_trace(trace_id, group, project_id, use_judge)
            if d:
                out.append(d)
        window = llm_rules.evaluate_window(events)
        if window:
            d = self._build(window, events, project_id or next((e.get("project_id") for e in events), None), None, use_judge)
            if d:
                out.append(d)
        return sorted(out, key=lambda d: (d.cross_paradigm, d.confidence), reverse=True)
