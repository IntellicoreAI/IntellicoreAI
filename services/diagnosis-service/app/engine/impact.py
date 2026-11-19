from typing import Any
from ..models import Finding, SEVERITY_ORDER


def assess_impact(primary: Finding, findings: list[Finding], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Scope of the problem: how many events/paradigms are affected, plus LLM cost/tokens involved."""
    ids = {i for f in findings for i in f.event_ids}
    affected = [e for e in events if str(e.get("event_id")) in ids]
    cost = tokens = 0.0
    for e in affected:
        m = ((e.get("metadata") or {}).get("metrics") or {}) if isinstance(e.get("metadata"), dict) else {}
        cost += float(m.get("cost_usd") or 0)
        tokens += float(m.get("total_tokens") or 0)
    sev = max((f.severity for f in findings), key=lambda s: SEVERITY_ORDER.get(s, 0), default="low")
    return {
        "severity_level": sev,
        "affected_event_count": len(ids),
        "affected_paradigms": sorted({str(e.get("system_type", "")).lower() for e in affected}),
        "affected_traces": sorted({str(e.get("trace_id")) for e in affected if e.get("trace_id")}),
        "llm_cost_usd_involved": round(cost, 6),
        "llm_tokens_involved": int(tokens),
        "requires_human_review": SEVERITY_ORDER.get(sev, 0) >= SEVERITY_ORDER["high"],
    }
