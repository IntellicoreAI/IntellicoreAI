from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, ConfigDict

class LLMEvent(BaseModel):
    """Validated telemetry emitted for one LLM request."""
    model_config = ConfigDict(extra="forbid")
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_type: Literal["LLM"] = "LLM"
    project_id: str = Field(min_length=1, max_length=255)
    model_id: str | None = Field(default=None, max_length=255)
    trace_id: str = Field(min_length=1, max_length=255)
    span_id: str = Field(min_length=1, max_length=255)
    event_type: str = Field(default="llm.request", min_length=1, max_length=255)
    input: Any = None
    output: Any = None
    latency: float | None = Field(default=None, ge=0, description="Milliseconds")
    status: str = Field(default="success", min_length=1, max_length=50)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class EvaluateRequest(BaseModel):
    project_id: str = "demo"
    model_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    prompt: str = ""
    response: str = ""
    context: str | list[str] | None = None
    expected_answer: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] = Field(default_factory=list)
    prompt_type: str | None = None
    language: str | None = None
    locale: str | None = None
    environment: str | None = None
    region: str | None = None
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    expected_format: str | None = None
    citations: list[str] = Field(default_factory=list)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    throughput_tokens_per_second: float | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    status: str = "success"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class FeedbackRequest(BaseModel):
    event_id: str
    vote: Literal["up", "down"]
    issue_category: str | None = None
    feedback_text: str | None = None


class ChatRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=255)
    prompt: str = Field(min_length=1)
    model: str | None = None
    system_prompt: str | None = None
    context: str | list[str] | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    trace_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class DriftRequest(BaseModel):
    project_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    reference: list[float] = Field(min_length=5)
    current: list[float] = Field(min_length=5)
    threshold: float = Field(default=2.0, gt=0)

class SemanticRequest(BaseModel):
    reference_text: str = Field(min_length=1)
    candidate_text: str = Field(min_length=1)
    model: str | None = None
