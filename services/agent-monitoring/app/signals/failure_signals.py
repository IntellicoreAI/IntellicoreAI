from typing import Any


def detect_failure_signals(
    tool_results: list[dict[str, Any]] | None = None,
    status: str = "completed",
    latency_ms: float | None = None,
    latency_threshold_ms: float = 1000.0,
) -> dict[str, Any]:
    """
    Detect observable failure signals from an Agentic execution.

    Signals include tool failures, execution failures, and
    unusually high execution latency.
    """

    tool_results = tool_results or []

    failed_tools = [
        result.get("tool")
        for result in tool_results
        if result.get("status") == "failed"
    ]

    signals: list[str] = []

    if failed_tools:
        signals.append("tool_failure")

    if status == "failed":
        signals.append("execution_failure")

    if latency_ms is not None and latency_ms > latency_threshold_ms:
        signals.append("latency_spike")

    return {
        "failure_detected": bool(signals),
        "signals": signals,
        "failed_tools": failed_tools,
        "latency_ms": latency_ms,
        "latency_threshold_ms": latency_threshold_ms,
    }