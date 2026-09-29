from app.loop_detector.detector import detect_repeated_nodes


def test_repeated_node_loop():

    trajectory = [
        {"node": "tool_node"},
        {"node": "tool_node"},
        {"node": "tool_node"},
    ]

    result = detect_repeated_nodes(trajectory)

    assert result["loop_detected"] is True
    assert result["loop_type"] == "repeated_node"


def test_alternating_cycle():

    trajectory = [
        {"node": "agent_llm"},
        {"node": "tool_node"},
        {"node": "agent_llm"},
        {"node": "tool_node"},
    ]

    result = detect_repeated_nodes(trajectory)

    assert result["loop_detected"] is True
    assert result["loop_type"] == "alternating_cycle"


def test_no_loop():

    trajectory = [
        {"node": "agent_llm"},
        {"node": "tool_node"},
        {"node": "execute_task"},
    ]

    result = detect_repeated_nodes(trajectory)

    assert result["loop_detected"] is False