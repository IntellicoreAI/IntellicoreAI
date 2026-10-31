import json, re

def evaluate_response_quality(response: str, expected_format: str | None = None, citations: list[str] | None = None) -> dict:
    words = re.findall(r"\b\w+\b", response)
    nonempty = bool(response.strip())
    unique_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
    completeness = round(min(1.0, len(words) / 30), 4) if nonempty else 0.0
    coherence = round(0.7 if nonempty and len(words) >= 5 else 0.35 if nonempty else 0.0, 4)
    format_valid = None
    if expected_format and expected_format.lower() == "json":
        try: json.loads(response); format_valid = True
        except (ValueError, TypeError): format_valid = False
    citation_coverage = None
    if citations is not None:
        citation_coverage = round(min(1.0, len(citations) / max(1, len(re.findall(r"[.!?]", response)))), 4)
    # These are deliberately labeled as proxies; no semantic classifier is bundled.
    toxicity_terms = ["kill yourself", "racial slur", "i will hurt you"]
    toxicity_flag = any(t in response.lower() for t in toxicity_terms)
    bias_terms = ["all women are", "all men are", "people of this race are"]
    bias_flag = any(t in response.lower() for t in bias_terms)
    return {
      "score": round((0.35 if nonempty else 0) + (0.25 if len(words) >= 5 else 0) + (0.2 if unique_ratio >= .35 else 0) + (0.1 if coherence >= .7 else 0) + (0.1 if len(response) <= 12000 else 0), 4),
      "label": "acceptable" if nonempty and len(words) >= 5 else "review",
      "word_count": len(words), "response_length_chars": len(response),
      "completeness_proxy": completeness, "coherence_proxy": coherence,
      "toxicity_rule_triggered": toxicity_flag, "bias_rule_triggered": bias_flag,
      "json_format_valid": format_valid, "citation_coverage_proxy": citation_coverage,
      "faithfulness": {"score": None, "label": "not_evaluable_without_semantic_judge"},
      "financial_term_correctness": {"score": None, "label": "requires_financial_reference_data"},
      "regulatory_accuracy": {"score": None, "label": "requires_versioned_regulatory_rules_and_expert_validation"},
      "reason": "Completeness/coherence/toxicity/bias/citation checks are lightweight proxies, not certified semantic evaluations"}
