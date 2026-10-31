"""Runtime validation schemas for IntelliCore diagnosis signals."""

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


SIGNAL_SCHEMA_VERSION = "1.0"

SignalType = Literal[
    "data_quality",
    "data_drift",
    "prediction_drift",
    "performance_degradation",
    "other",
]

SignalSeverity = Literal[
    "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"
]

SignalStatus = Literal[
    "detected", "failed", "warning"
]


class ModelInformation(BaseModel):
    """Information identifying the monitored ML model."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)


class MLDiagnosisSignal(BaseModel):
    """A structured ML finding with evidence and source traceability."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    signal_id: str = Field(min_length=1)
    created_at: datetime
    source: str = Field(min_length=1)

    signal_type: SignalType
    status: SignalStatus
    severity: SignalSeverity

    project_id: str = Field(min_length=1)
    environment: Literal["development", "staging", "production"]

    model: ModelInformation

    issue: str = Field(min_length=1)
    investigation_area: str = Field(min_length=1)

    evidence: dict[str, Any]

    source_alert_id: str | None
    diagnosis_id: str = Field(min_length=1)

    root_cause: str | None = None


class MLDiagnosisSignalDocument(BaseModel):
    """Validates the complete collection of generated ML signals."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    generated_at: datetime
    source_diagnosis_id: str = Field(min_length=1)
    signal_count: int = Field(ge=0)
    signals: list[MLDiagnosisSignal]

    @model_validator(mode="after")
    def validate_signal_count(self):
        if self.signal_count != len(self.signals):
            raise ValueError(
                "signal_count does not match the number of signals."
            )

        return self
