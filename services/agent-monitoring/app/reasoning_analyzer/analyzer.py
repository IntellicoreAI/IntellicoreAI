from typing import Any


def analyze_execution_reasoning(
    trajectory: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    """
    Analyze structured agent execution evidence.

    This does not capture private chain-of-thought.
    It summarizes observable execution behavior.
    """

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

    if status == "completed":
        execution_decision = "task_completed"
    elif failed_tools:
        execution_decision = "tool_failure_detected"
    else:
        execution_decision = "execution_incomplete"

    return {
        "executed_nodes": executed_nodes,
        "executed_tools": executed_tools,
        "failed_tools": failed_tools,
        "execution_decision": execution_decision,
        "execution_status": status,
    }