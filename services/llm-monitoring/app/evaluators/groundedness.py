from typing import Any
import re

def _terms(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(x) > 2}

def score_groundedness(response: str, context: str | list[str] | None) -> dict:
    """Transparent lexical evidence proxy; not a claim of semantic truth."""
    if not context:
        return {"score": None, "label": "not_evaluable", "reason": "No retrieved context supplied"}
    ctx = " ".join(context) if isinstance(context, list) else context
    response_terms, context_terms = _terms(response), _terms(ctx)
    score = round(len(response_terms & context_terms) / max(1, len(response_terms)), 4)
    return {"score": score, "label": "grounded" if score >= .55 else "weakly_grounded",
            "reason": "Token overlap proxy; validate important claims against source evidence"}
