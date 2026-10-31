from typing import Any


def generate_agent_recommendation(
    diagnosis: dict[str, Any],
    failure_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate an actionable recommendation from Agentic AI diagnosis
    and observable failure signals.
    """

    root_cause = diagnosis.get("root_cause")
    severity = diagnosis.get("severity", "low")
    signals = failure_analysis.get("signals", [])

    recommendation: dict[str, Any] = {
        "recommendation_detected": False,
        "priority": "low",
        "action": None,
        "reason": None,
        "evidence": [],
    }

    if root_cause == "tool_execution_failure":
        recommendation.update(
            {
                "recommendation_detected": True,
                "priority": "high",
                "action": "retry_or_validate_tool",
                "reason": (
                    "A tool execution failure was detected. "
                    "Retry the tool when appropriate and validate "
                    "tool availability before continuing the workflow."
                ),
            }
        )

    elif root_cause == "repeated_execution_loop":
        recommendation.update(
            {
                "recommendation_detected": True,
                "priority": "high",
                "action": "terminate_or_limit_loop",
                "reason": (
                    "Repeated execution patterns were detected. "
                    "Apply an execution limit or terminate the loop "
                    "before additional unnecessary tool calls."
                ),
            }
        )

    elif root_cause == "agent_execution_failure":
        recommendation.update(
            {
                "recommendation_detected": True,
                "priority": "high",
                "action": "investigate_execution_failure",
                "reason": (
                    "The agent execution failed without a more "
                    "specific root cause. Inspect the execution "
                    "trace and failure signals."
                ),
            }
        )

    elif root_cause == "trace_execution_anomaly":
        recommendation.update(
            {
                "recommendation_detected": True,
                "priority": "medium",
                "action": "inspect_execution_trace",
                "reason": (
                    "An execution trace anomaly was detected. "
                    "Review the affected spans and execution sequence."
                ),
            }
        )

    elif root_cause == "tool_failure_detected_by_agent":
        recommendation.update(
            {
                "recommendation_detected": True,
                "priority": "medium",
                "action": "review_tool_failure",
                "reason": (
                    "The execution analysis identified a tool failure. "
                    "Review the tool result and determine whether retry "
                    "or escalation is appropriate."
                ),
            }
        )

    else:
        recommendation.update(
            {
                "recommendation_detected": False,
                "priority": "low",
                "action": "continue_monitoring",
                "reason": (
                    "No actionable Agentic execution failure was detected. "
                    "Continue monitoring the execution."
                ),
            }
        )

    recommendation["evidence"] = [
        {
            "root_cause": root_cause,
            "severity": severity,
            "signals": signals,
        }
    ]

    return recommendation