import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[2]

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.analytics.analyzer import calculate_agent_metrics


def test_agent_metrics():
    trajectory = [
        {"node": "agent_llm"},
        {"node": "execute_tools"},
        {"node": "execute_task"},
    ]

    tool_calls = [
        {"tool": "check_transaction_status"},
    ]

    tool_results = [
        {
            "tool": "check_transaction_status",
            "status": "completed",
        }
    ]

    telemetry_events = [
        {
            "name": "agent_llm",
            "status": "completed",
            "latency_ms": 100.0,
        },
        {
            "name": "tool_execution",
            "status": "completed",
            "latency_ms": 50.0,
        },
    ]

    metrics = calculate_agent_metrics(
        trajectory=trajectory,
        tool_calls=tool_calls,
        tool_results=tool_results,
        telemetry_events=telemetry_events,
        status="completed",
    )

    assert metrics["execution_success"] is True
    assert metrics["trajectory_length"] == 3
    assert metrics["tool_call_count"] == 1
    assert metrics["tool_result_count"] == 1
    assert metrics["successful_tool_results"] == 1
    assert metrics["tool_failure_count"] == 0
    assert metrics["execution_failure_count"] == 0
    assert metrics["telemetry_event_count"] == 2
    assert metrics["total_telemetry_latency_ms"] == 150.0


def test_agent_metrics_with_tool_failure():
    metrics = calculate_agent_metrics(
        trajectory=[
            {"node": "agent_llm"},
            {"node": "execute_tools"},
        ],
        tool_calls=[
            {"tool": "check_transaction_status"},
        ],
        tool_results=[
            {
                "tool": "check_transaction_status",
                "status": "failed",
            }
        ],
        telemetry_events=[
            {
                "name": "tool_execution",
                "status": "failed",
                "latency_ms": 75.0,
            }
        ],
        status="failed",
    )

    assert metrics["execution_success"] is False
    assert metrics["tool_failure_count"] == 1
    assert metrics["execution_failure_count"] == 1