from typing import Any, Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict, total=False):
    """State carried through an IntelliCore LangGraph execution."""

    task: str
    input_data: dict[str, Any]

    # LangGraph accumulates messages from each node instead of replacing them.
    messages: Annotated[list[AnyMessage], add_messages]

    current_node: str

    trajectory: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # Telemetry collected across the complete Agentic execution.
    telemetry_events: list[dict[str, Any]]

    memory: list[dict[str, Any]]
    memory_analysis: dict[str, Any]

    trace_analysis: dict[str, Any]

    output: Any
    error: str | None
    status: str

    trace_id: str
    span_id: str
    execution_start_time: float

    attempt_count: int
    max_attempts: int

    loop_analysis: dict[str, Any]
    failure_analysis: dict[str, Any]
    diagnosis: dict[str, Any]
    recommendation: dict[str, Any]

    # Aggregated Agentic execution metrics.
    analytics: dict[str, Any]

    event: dict[str, Any]