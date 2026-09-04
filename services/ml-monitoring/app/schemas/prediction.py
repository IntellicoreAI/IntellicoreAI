from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionEvent(BaseModel):
    project_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    environment: str = Field(min_length=1)

    timestamp: datetime

    features: dict[str, Any]

    prediction: Any
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    actual_label: Any | None = None

    latency_ms: float = Field(ge=0.0)
