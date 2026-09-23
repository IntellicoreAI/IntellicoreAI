from typing import Any


def diagnose_agent_execution(
    failure_analysis: dict[str, Any],
    tool_results: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    reasoning_analysis: dict[str, Any],
    loop_analysis: dict[str, Any],
    trace_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Diagnose the primary reliability issue observed during
    an Agentic AI execution.

    The diagnosis is based only on observable execution evidence.
    It does not attempt to reconstruct private model reasoning.
    """

    signals = failure_analysis.get("signals", [])

    failed_tools = failure_analysis.get("failed_tools", [])

    diagnosis: dict[str, Any] = {
        "diagnosis_detected": False,
        "root_cause": None,
        "category": None,
        "severity": "low",
        "explanation": None,
        "evidence": [],
    }

    # 1. Tool failure
    if "tool_failure" in signals:
        diagnosis.update(
            {
                "diagnosis_detected": True,
                "root_cause": "tool_execution_failure",
                "category": "tool",
                "severity": "high",
                "explanation": (
                    "The agent execution encountered a failure "
                    "while executing a tool."
                ),
            }
        )

        if failed_tools:
            diagnosis["evidence"].append(
                {
                    "type": "failed_tool",
                    "tools": failed_tools,
                }
            )

        for tool_result in tool_results:
            if tool_result.get("status") == "failed":
                diagnosis["evidence"].append(
                    {
                        "type": "tool_result",
                        "tool": tool_result.get("tool"),
                        "error": tool_result.get("error"),
                    }
                )

    # 2. Loop detection
    elif loop_analysis.get("loop_detected"):
        diagnosis.update(
            {
                "diagnosis_detected": True,
                "root_cause": "repeated_execution_loop",
                "category": "loop",
                "severity": "high",
                "explanation": (
                    "The agent trajectory contains repeated "
                    "execution patterns indicating a possible loop."
                ),
            }
        )

        diagnosis["evidence"].append(
            {
                "type": "loop_analysis",
                "analysis": loop_analysis,
            }
        )

    # 3. Execution failure without a specific tool failure
    elif "execution_failure" in signals:
        diagnosis.update(
            {
                "diagnosis_detected": True,
                "root_cause": "agent_execution_failure",
                "category": "execution",
                "severity": "high",
                "explanation": (
                    "The agent execution failed, but the available "
                    "signals do not identify a more specific root cause."
                ),
            }
        )

    # 4. Trace-level anomaly
    elif trace_analysis.get("trace_failed"):
        diagnosis.update(
            {
                "diagnosis_detected": True,
                "root_cause": "trace_execution_anomaly",
                "category": "trace",
                "severity": "medium",
                "explanation": (
                    "The execution trace contains an anomaly "
                    "that requires further investigation."
                ),
            }
        )

        diagnosis["evidence"].append(
            {
                "type": "trace_analysis",
                "analysis": trace_analysis,
            }
        )

    # 5. Reasoning/execution decision anomaly
    elif reasoning_analysis.get("decision") == "tool_failure_detected":
        diagnosis.update(
            {
                "diagnosis_detected": True,
                "root_cause": "tool_failure_detected_by_agent",
                "category": "reasoning",
                "severity": "medium",
                "explanation": (
                    "The execution analysis identified a tool "
                    "failure during the agent trajectory."
                ),
            }
        )

        diagnosis["evidence"].append(
            {
                "type": "reasoning_analysis",
                "analysis": reasoning_analysis,
            }
        )

    # 6. No detected failure
    else:
        diagnosis.update(
            {
                "diagnosis_detected": False,
                "root_cause": "no_failure_detected",
                "category": "none",
                "severity": "low",
                "explanation": (
                    "No actionable Agentic execution failure "
                    "was detected from the available telemetry."
                ),
            }
        )

    diagnosis["evidence"].append(
        {
            "type": "trajectory",
            "trajectory_length": len(trajectory),
        }
    )

    return diagnosis