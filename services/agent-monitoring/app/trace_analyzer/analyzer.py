from typing import Any


def analyze_trace(
    trajectory: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Analyze observable execution trace information.

    This summarizes nodes, tools, statuses, and execution steps.
    It does not inspect private chain-of-thought.
    """

    trajectory = trajectory or []
    tool_calls = tool_calls or []
    tool_results = tool_results or []

    executed_nodes = [
        step.get("node")
        for step in trajectory
        if step.get("node")
    ]

    executed_tools = [
        call.get("tool")
        for call in tool_calls
        if call.get("tool")
    ]

    failed_tools = [
        result.get("tool")
        for result in tool_results
        if result.get("status") == "failed"
    ]

    completed_tools = [
        result.get("tool")
        for result in tool_results
        if result.get("status") == "completed"
    ]

    return {
        "trace_available": bool(trajectory),
        "step_count": len(trajectory),
        "executed_nodes": executed_nodes,
        "executed_tools": executed_tools,
        "completed_tools": completed_tools,
        "failed_tools": failed_tools,
        "trace_status": (
            "failed"
            if failed_tools
            else "completed"
            if trajectory
            else "empty"
        ),
    }