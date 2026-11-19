def test_agentic_failure_diagnosis(monkeypatch):
    """Verify tool failure is detected and diagnosed correctly."""

    monkeypatch.setattr(
        "app.execution_monitor.graph.publish_event",
        lambda event: {"status": "published"},
    )

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
