from ..models import Finding, SEVERITY_ORDER

# Cross-paradigm explanations outrank single-paradigm ones: they explain more of the trace.
_PARADIGM_RANK = {"cross": 2, "agent": 1, "llm": 1}


def rank_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (_PARADIGM_RANK.get(f.paradigm, 0), SEVERITY_ORDER.get(f.severity, 0), f.base_confidence), reverse=True)


def select_root_cause(findings: list[Finding]) -> tuple[Finding | None, list[Finding]]:
    """Return (primary, contributing). A critical single-paradigm finding can outrank a weak correlation-only one."""
    ranked = rank_findings(findings)
    if not ranked:
        return None, []
    primary = ranked[0]
    if primary.root_cause == "llm_issue_coincides_with_agent_failure":
        stronger = [f for f in ranked[1:] if f.paradigm != "cross" and SEVERITY_ORDER.get(f.severity, 0) >= SEVERITY_ORDER["high"]]
        if stronger:
            primary = stronger[0]
    return primary, [f for f in ranked if f is not primary]
