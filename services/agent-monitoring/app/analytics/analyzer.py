from typing import Any


def calculate_agent_metrics(
    trajectory: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    telemetry_events: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    """Calculate observable execution metrics for an Agentic AI run."""

    tool_call_count = len(tool_calls)
    tool_result_count = len(tool_results)

    tool_failure_count = sum(
        1
        for result in tool_results
        if result.get("status") == "failed"
    )

    execution_failure_count = sum(
        1
        for event in telemetry_events
        if event.get("status") == "failed"
    )

    latencies = [
        event.get("latency_ms")
        for event in telemetry_events
        if isinstance(event.get("latency_ms"), (int, float))
    ]

    total_telemetry_latency = sum(latencies)

    successful_tool_results = sum(
        1
        for result in tool_results
        if result.get("status") == "completed"
    )

    if status == "completed":
        execution_success = True
    else:
        execution_success = False

    return {
        "execution_success": execution_success,
        "trajectory_length": len(trajectory),
        "tool_call_count": tool_call_count,
        "tool_result_count": tool_result_count,
        "successful_tool_results": successful_tool_results,
        "tool_failure_count": tool_failure_count,
        "execution_failure_count": execution_failure_count,
        "telemetry_event_count": len(telemetry_events),
        "total_telemetry_latency_ms": total_telemetry_latency,
    }