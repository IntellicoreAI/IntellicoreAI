import sys
from pathlib import Path

from langchain_core.messages import AIMessage

SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.execution_monitor.graph import execute_tools, execute_task


def build_failure_state():
    """Build a deterministic agent state containing a failed tool request."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_transaction_status",
                "args": {"transaction_id": "INVALID-FAIL"},
                "id": "test-tool-call-001",
                "type": "tool_call",
            }
        ],
    )

    return {
        "task": "Check the status of transaction INVALID-FAIL",
        "input_data": {},
        "messages": [ai_message],
        "trajectory": [],
        "tool_calls": [],
        "tool_results": [],
        "memory": [],
        "attempt_count": 0,
        "max_attempts": 3,
        "trace_id": "test-trace-001",
        "span_id": "test-span-001",
        "status": "running",
    }


def test_agentic_failure_diagnosis():
    """Verify tool failure is detected and diagnosed correctly."""
    state = build_failure_state()

    state = execute_tools(state)
    state = execute_task(state)

    assert state["status"] == "failed"

    diagnosis = state["diagnosis"]

    assert diagnosis["diagnosis_detected"] is True
    assert diagnosis["root_cause"] == "tool_execution_failure"
    assert diagnosis["category"] == "tool"
    assert diagnosis["severity"] == "high"

    event = state["event"]

    assert event["event_type"] == "agent_execution"
    assert event["status"] == "failed"

    failure_analysis = state["failure_analysis"]

    assert failure_analysis["failure_detected"] is True
    assert "tool_failure" in failure_analysis["signals"]
    assert "execution_failure" in failure_analysis["signals"]