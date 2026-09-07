from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State carried through an IntelliCore LangGraph execution."""

    task: str
    input_data: dict[str, Any]

    current_node: str
    trajectory: list[dict[str, Any]]

    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    output: Any

    error: str | None
    status: str

    trace_id: str
    span_id: str

    event: dict[str, Any]