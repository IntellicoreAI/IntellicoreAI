from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State carried through an IntelliCore LangGraph execution."""

    task: str
    input_data: dict[str, Any]

    current_node: str
    trajectory: list[dict[str, Any]]

    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # Observable memory entries used during agent execution.
    memory: list[dict[str, Any]]

    # Result produced by the Agentic memory analyzer.
    memory_analysis: dict[str, Any]

    # Result produced by the Agentic trace analyzer.
    trace_analysis: dict[str, Any]

    output: Any

    error: str | None
    status: str

    trace_id: str
    span_id: str

    # Number of times the agent has attempted execution.
    attempt_count: int

    # Maximum number of allowed execution attempts.
    max_attempts: int

    # Result produced by Agentic monitoring components.
    loop_analysis: dict[str, Any]

    # Result produced by the Agentic failure-signal detector.
    failure_analysis: dict[str, Any]

    event: dict[str, Any]