import re

def evaluate_relevance(prompt: str, response: str) -> dict:
    p = {w for w in re.findall(r"[a-zA-Z0-9]+", prompt.lower()) if len(w) > 2}
    r = {w for w in re.findall(r"[a-zA-Z0-9]+", response.lower()) if len(w) > 2}
    score = round(len(p & r) / max(1, len(p)), 4) if p else None
    return {"score": score, "label": ("relevant" if score >= .25 else "possibly_irrelevant") if score is not None else "not_evaluable",
            "reason": "Prompt/response keyword coverage proxy; semantic relevance requires a model-based evaluator"}
