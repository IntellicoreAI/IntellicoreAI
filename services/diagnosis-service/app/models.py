from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class Finding(BaseModel):
    """One rule hit. Rules never decide the final diagnosis; the engine ranks findings."""
    rule_id: str
    paradigm: str                       # llm | agent | cross
    root_cause: str
    category: str
    severity: str = "medium"
    base_confidence: float = Field(0.6, ge=0, le=1)
    title: str = ""
    explanation: str = ""
    trace_id: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    recommended_action: str | None = None


class Diagnosis(BaseModel):
    diagnosis_id: str
    project_id: str | None = None
    trace_id: str | None = None
    paradigms: list[str] = Field(default_factory=list)
    root_cause: str
    category: str
    severity: str
    confidence: float
    impact: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    contributing_findings: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    cross_paradigm: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))


class DiagnoseRequest(BaseModel):
    project_id: str | None = None
    trace_id: str | None = None
    events: list[dict[str, Any]] | None = None   # explicit events instead of reading the DB
    use_judge: bool = False
    persist: bool = True
