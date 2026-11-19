import json
from typing import Any
from ..models import Finding


def merge_evidence(findings: list[Finding], extra: list[dict[str, Any]] | None = None, limit: int = 40) -> list[dict[str, Any]]:
    """De-duplicated evidence from all findings (+ optional extras such as LLM-judge verdicts)."""
    seen, out = set(), []
    for item in [e for f in findings for e in f.evidence] + (extra or []):
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:limit]


def llm_evaluation_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the LLM evaluation results (what the Decision Pipeline consumes as LLM evidence)."""
    return [e for e in evidence if e.get("source") == "llm" and e.get("type") in {"evaluation", "risk_flags", "latency", "error", "window_stat"}]
