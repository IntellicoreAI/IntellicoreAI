"""Optional LLM-as-judge used as *extra evidence*, never as the sole basis of a diagnosis.

Calls an OpenAI-compatible chat endpoint (JUDGE_* env vars, falling back to LLM_*). When no API key is
configured, or the call fails, the verdict is simply `available=False` and rule-based diagnosis proceeds.
"""
import json
import os
import re
import urllib.request
from typing import Any, Callable

from .schemas import JudgeRequest, JudgeVerdict

SYSTEM_PROMPT = (
    "You are a strict QA judge for an LLM monitoring system. Given a prompt, optional context and a response, decide whether "
    "the suspected issue is present. Reply with ONLY JSON: "
    '{"issue_confirmed": true|false, "score": <0..1 severity>, "reasoning": "<one or two sentences>"}'
)


def _config() -> dict[str, Any]:
    return {
        "key": os.getenv("JUDGE_LLM_API_KEY") or os.getenv("LLM_API_KEY", ""),
        "base": (os.getenv("JUDGE_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        "model": os.getenv("JUDGE_LLM_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
        "timeout": float(os.getenv("JUDGE_TIMEOUT_SECONDS", "20")),
    }


def _http_transport(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_verdict_text(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if "issue_confirmed" not in data:
        return None
    score = data.get("score")
    return {"issue_confirmed": bool(data["issue_confirmed"]),
            "score": max(0.0, min(1.0, float(score))) if isinstance(score, (int, float)) else None,
            "reasoning": str(data.get("reasoning", ""))[:500]}


def judge(req: JudgeRequest, transport: Callable | None = None) -> JudgeVerdict:
    cfg = _config()
    if not cfg["key"] and transport is None:
        return JudgeVerdict(available=False, note="No JUDGE_LLM_API_KEY/LLM_API_KEY configured; rule-based diagnosis only.")
    user = (f"Suspected issue: {req.suspected_issue or 'general quality'}\n\nPrompt:\n{req.prompt[:3000]}\n\n"
            f"Context:\n{(req.context or 'none')[:3000]}\n\nResponse:\n{req.response[:3000]}")
    body = {"model": cfg["model"], "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}"}
    try:
        raw = (transport or _http_transport)(f"{cfg['base']}/chat/completions", headers, body, cfg["timeout"])
        parsed = parse_verdict_text(raw["choices"][0]["message"]["content"])
    except Exception as exc:  # judge is best-effort evidence
        return JudgeVerdict(available=False, note=f"Judge call failed: {type(exc).__name__}")
    if parsed is None:
        return JudgeVerdict(available=False, note="Judge returned an unparseable verdict.")
    return JudgeVerdict(available=True, model=cfg["model"], **parsed)


def judge_events(events: list[dict[str, Any]], transport: Callable | None = None) -> dict[str, Any]:
    """Judge the most concerning LLM event of a trace. Used as DiagnosisEngine(judge=judge_events)."""
    def risk(e):
        md = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
        return len(md.get("risk_flags") or [])
    target = max(events, key=risk, default=None)
    if not target:
        return {"available": False}
    md = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
    flags = md.get("risk_flags") or []
    verdict = judge(JudgeRequest(prompt=str(target.get("input") or ""), response=str(target.get("output") or ""),
                                 context=md.get("context") if isinstance(md.get("context"), str) else None,
                                 suspected_issue=flags[0] if flags else None), transport)
    return {**verdict.model_dump(), "event_id": str(target.get("event_id"))}
