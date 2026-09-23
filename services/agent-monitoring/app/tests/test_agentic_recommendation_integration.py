import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[2]

if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.execution_monitor.graph import execute_task


def test_execute_task_generates_recommendation():
    state = {
        "task": "Check transaction status",
        "messages": [],
        "trajectory": [],
        "tool_calls": [],
        "tool_results": [
            {
                "tool": "check_transaction_status",
                "output": None,
                "status": "failed",
                "latency_ms": 5.0,
                "error": "Transaction service failed.",
                "trace_id": "trace-test",
                "span_id": "span-test",
                "tool_call_id": "call-test",
            }
        ],
        "memory": [],
        "trace_id": "trace-test",
        "attempt_count": 0,
        "max_attempts": 3,
        "execution_start_time": None,
    }

    result = execute_task(state)

    assert result["failure_analysis"]["failure_detected"] is True
    assert result["diagnosis"]["diagnosis_detected"] is True

    recommendation = result["recommendation"]

    assert recommendation["recommendation_detected"] is True
    assert recommendation["priority"] == "high"
    assert recommendation["action"] == "retry_or_validate_tool"

    event = result["event"]

    assert "recommendation" in event["metadata"]
    assert (
        event["metadata"]["recommendation"]["action"]
        == "retry_or_validate_tool"
    )