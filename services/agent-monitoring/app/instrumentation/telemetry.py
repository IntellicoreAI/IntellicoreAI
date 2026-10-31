import time
import uuid
from typing import Any


class AgentTelemetry:
    """Capture observable telemetry for an Agentic AI execution."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.events: list[dict[str, Any]] = []

    def start_span(self, name: str) -> dict[str, Any]:
        """Start an observable span for a node or tool."""

        return {
            "span_id": str(uuid.uuid4()),
            "name": name,
            "trace_id": self.trace_id,
            "start_time": time.perf_counter(),
        }

    def end_span(
        self,
        span: dict[str, Any],
        status: str = "completed",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Complete a span and record its telemetry."""

        latency_ms = (
            time.perf_counter() - span["start_time"]
        ) * 1000

        event = {
            "span_id": span["span_id"],
            "trace_id": self.trace_id,
            "name": span["name"],
            "status": status,
            "latency_ms": latency_ms,
            "error": error,
            "metadata": metadata or {},
        }

        self.events.append(event)

        return event

    def record_tool(
        self,
        tool_name: str,
        status: str,
        latency_ms: float,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record telemetry for a tool execution."""

        event = {
            "span_id": str(uuid.uuid4()),
            "trace_id": self.trace_id,
            "name": tool_name,
            "type": "tool",
            "status": status,
            "latency_ms": latency_ms,
            "error": error,
            "metadata": metadata or {},
        }

        self.events.append(event)

        return event

    def record_node(
        self,
        node_name: str,
        status: str,
        latency_ms: float,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record telemetry for an Agentic node execution."""

        event = {
            "span_id": str(uuid.uuid4()),
            "trace_id": self.trace_id,
            "name": node_name,
            "type": "node",
            "status": status,
            "latency_ms": latency_ms,
            "error": error,
            "metadata": metadata or {},
        }

        self.events.append(event)

        return event