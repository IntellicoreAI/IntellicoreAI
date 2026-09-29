import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.reasoning_analyzer.analyzer import analyze_execution_reasoning


def test_completed_execution():
    result = analyze_execution_reasoning(
        trajectory=[
            {"node": "agent_llm"},
            {"node": "tool_node"},
            {"node": "execute_task"},
        ],
        tool_calls=[
            {"tool": "check_transaction_status"},
        ],
        tool_results=[
            {
                "tool": "check_transaction_status",
                "status": "completed",
            }
        ],
        status="completed",
    )

    assert result["executed_nodes"] == [
        "agent_llm",
        "tool_node",
        "execute_task",
    ]
    assert result["executed_tools"] == [
        "check_transaction_status",
    ]
    assert result["failed_tools"] == []
    assert result["execution_decision"] == "task_completed"
    assert result["execution_status"] == "completed"


def test_failed_tool_execution():
    result = analyze_execution_reasoning(
        trajectory=[
            {"node": "agent_llm"},
            {"node": "tool_node"},
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
        status="failed",
    )

    assert result["failed_tools"] == [
        "check_transaction_status",
    ]
    assert result["execution_decision"] == "tool_failure_detected"


def test_incomplete_execution():
    result = analyze_execution_reasoning(
        trajectory=[
            {"node": "agent_llm"},
        ],
        tool_calls=[],
        tool_results=[],
        status="incomplete",
    )

    assert result["executed_nodes"] == ["agent_llm"]
    assert result["executed_tools"] == []
    assert result["failed_tools"] == []
    assert result["execution_decision"] == "execution_incomplete"