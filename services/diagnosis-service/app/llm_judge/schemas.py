from pydantic import BaseModel, Field


class JudgeRequest(BaseModel):
    prompt: str
    response: str
    context: str | None = None
    suspected_issue: str | None = None   # e.g. hallucination_risk


class JudgeVerdict(BaseModel):
    available: bool
    issue_confirmed: bool | None = None
    score: float | None = Field(default=None, ge=0, le=1)   # 0 = fine, 1 = severe problem
    reasoning: str = ""
    model: str | None = None
    note: str | None = None
