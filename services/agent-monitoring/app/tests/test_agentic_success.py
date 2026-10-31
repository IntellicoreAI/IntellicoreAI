import sys
from pathlib import Path

from langchain_core.messages import AIMessage

SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.execution_monitor.graph import execute_tools, execute_task


def build_success_state():
    """Build a deterministic agent state containing a successful tool request."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_transaction_status",
                "args": {"transaction_id": "TXN-001"},
                "id": "test-tool-call-success-001",
                "type": "tool_call",
            }
        ],
    )

    return {
        "task": "Check the status of transaction TXN-001",
        "input_data": {},
        "messages": [ai_message],
        "trajectory": [],
        "tool_calls": [],
        "tool_results": [],
        "memory": [],
        "attempt_count": 0,
        "max_attempts": 3,
        "trace_id": "test-success-trace-001",
        "span_id": "test-success-span-001",
        "status": "running",
    }


def test_agentic_success_execution():
    """Verify successful tool execution produces a completed event."""
    state = build_success_state()

    state = execute_tools(state)
    state = execute_task(state)

    assert state["status"] == "completed"

    event = state["event"]

    assert event["event_type"] == "agent_execution"
    assert event["status"] == "completed"
    assert event["error"] is None

    diagnosis = state["diagnosis"]

    assert diagnosis["diagnosis_detected"] is False
    assert diagnosis["root_cause"] == "no_failure_detected"
    assert diagnosis["category"] == "none"