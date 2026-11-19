
from app.execution_monitor.graph import route_after_tools


def test_routes_to_execution_when_tool_limit_is_reached():
    state = {
        "trajectory": [],
        "tool_execution_count": 3,
        "max_attempts": 3,
    }

    assert route_after_tools(state) == "execute"


def test_routes_to_execution_when_loop_is_detected():
    state = {
        "trajectory": [
            {"node": "tool_node"},
            {"node": "agent_llm"},
            {"node": "tool_node"},
            {"node": "agent_llm"},
            {"node": "tool_node"},
        ],
        "tool_execution_count": 1,
        "max_attempts": 3,
    }

    assert route_after_tools(state) == "execute"


def test_continues_to_llm_when_limit_not_reached():
    state = {
        "trajectory": [],
        "tool_execution_count": 1,
        "max_attempts": 3,
    }

    assert route_after_tools(state) == "agent_llm"
