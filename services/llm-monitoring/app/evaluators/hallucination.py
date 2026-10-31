from .groundedness import score_groundedness

def evaluate_hallucination(response: str, context):
    grounded = score_groundedness(response, context)
    if grounded["score"] is None:
        return {"score": None, "label": "not_evaluable", "reason": grounded["reason"]}
    score = round(1.0 - grounded["score"], 4)
    return {"score": score, "label": "high_risk" if score >= .65 else "low_risk",
            "reason": "Heuristic risk indicator derived from lexical grounding; not a factuality guarantee"}
