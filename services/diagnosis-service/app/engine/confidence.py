from ..models import Finding


def score_confidence(primary: Finding, all_findings: list[Finding], judge_agrees: bool | None = None) -> float:
    """Base rule confidence, corroborated by independent paradigms/rules, rounded to 2 dp."""
    conf = primary.base_confidence
    paradigms = {f.paradigm for f in all_findings if f.paradigm != "cross"}
    if primary.paradigm == "cross":
        conf += 0.05 * max(0, len(paradigms) - 1)            # both LLM and agent evidence present
    corroborating = [f for f in all_findings if f is not primary and f.root_cause == primary.root_cause]
    conf += min(0.1, 0.05 * len(corroborating))
    if len(primary.evidence) >= 4:
        conf += 0.03
    if judge_agrees is True:
        conf += 0.05
    elif judge_agrees is False:
        conf -= 0.1
    return round(max(0.05, min(0.99, conf)), 2)
