import uuid

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..instrumentation.telemetry import AgentTelemetry
from ..llm.model import get_qwen3
from ..loop_detector.detector import detect_repeated_nodes
from ..memory_analyzer.analyzer import analyze_memory
from ..reasoning_analyzer.analyzer import analyze_execution_reasoning
from ..signals.event_schema import AgenticEvent
from ..signals.failure_signals import detect_failure_signals
from ..tool_monitor.tools import check_transaction_status
from ..trace_analyzer.analyzer import analyze_trace
from .state import AgentState


def agent_llm(state: AgentState) -> AgentState:
    """Run the real local Qwen3 agent model."""

    model = get_qwen3().bind_tools([check_transaction_status])

    messages = state.get("messages", [])

    if not messages:
        messages = [
            {
                "role": "user",
                "content": state.get("task", ""),
            }
        ]

    response = model.invoke(messages)

    return {
        "messages": messages + [response],
        "current_node": "agent_llm",
    }


def route_after_llm(state: AgentState) -> str:
    """Route the agent based on Qwen3's response."""

    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tool"

    return "execute"


def execute_tools(state: AgentState) -> AgentState:
    """
    Execute the real LangGraph ToolNode with IntelliCore telemetry.

    ToolNode converts tool exceptions into ToolMessage error results.
    IntelliCore inspects those results and records the tool execution
    as failed without executing the tool twice.
    """

    trace_id = state.get("trace_id") or str(uuid.uuid4())

    telemetry = AgentTelemetry(trace_id=trace_id)

    tool_node = ToolNode([check_transaction_status])

    tool_span = telemetry.start_span("tool_node")

    try:
        result = tool_node.invoke(state)
    except Exception as exc:
        result = {"messages": []}

        tool_status = "failed"
        tool_error = str(exc)

    else:
        tool_status = "completed"
        tool_error = None

        # ToolNode normally converts tool exceptions into ToolMessage
        # objects instead of raising them. Inspect the returned messages.
        for message in result.get("messages", []):
            message_status = getattr(message, "status", None)

            if message_status == "error":
                tool_status = "failed"
                tool_error = str(getattr(message, "content", ""))

                break

    tool_event = telemetry.end_span(
        tool_span,
        status=tool_status,
        error=tool_error,
        metadata={
            "node": "tool_node",
        },
    )

    messages = result.get("messages", [])

    tool_name = "unknown_tool"
    tool_call_id = None
    tool_output = None

    # Identify the actual tool call requested by Qwen3.
    for message in reversed(state.get("messages", [])):
        tool_calls = getattr(message, "tool_calls", [])

        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name", "unknown_tool")
            tool_call_id = tool_call.get("id")
            break

    # Identify the actual ToolNode result.
    for message in reversed(messages):
        if message.__class__.__name__ == "ToolMessage":
            tool_output = message.content

            if getattr(message, "name", None):
                tool_name = message.name

            if getattr(message, "tool_call_id", None):
                tool_call_id = message.tool_call_id

            if getattr(message, "status", None) == "error":
                tool_status = "failed"
                tool_error = str(message.content)

            break

    tool_results = state.get("tool_results", []).copy()

    tool_results.append(
        {
            "tool": tool_name,
            "output": tool_output,
            "status": tool_status,
            "latency_ms": tool_event["latency_ms"],
            "error": tool_error,
            "trace_id": trace_id,
            "span_id": tool_event["span_id"],
            "tool_call_id": tool_call_id,
        }
    )

    trajectory = state.get("trajectory", []).copy()

    trajectory.append(
        {
            "node": "tool_node",
            "status": tool_status,
            "tool": tool_name,
            "tool_status": tool_status,
            "tool_latency_ms": tool_event["latency_ms"],
            "trace_id": trace_id,
            "span_id": tool_event["span_id"],
        }
    )

    return {
        "messages": messages,
        "tool_results": tool_results,
        "trajectory": trajectory,
        "trace_id": trace_id,
        "current_node": "tool_node",
    }


def execute_task(state: AgentState) -> AgentState:
    """
    Finalize the real agent execution.

    Tool execution is performed by LangGraph ToolNode.
    This node observes the resulting messages and runs
    IntelliCore analysis and telemetry generation.
    """

    task = state.get("task", "")
    trajectory = state.get("trajectory", []).copy()
    tool_calls = state.get("tool_calls", []).copy()
    tool_results = state.get("tool_results", []).copy()
    memory = state.get("memory", []).copy()
    messages = state.get("messages", [])

    trace_id = state.get("trace_id") or str(uuid.uuid4())

    telemetry = AgentTelemetry(trace_id=trace_id)

    attempt_count = state.get("attempt_count", 0) + 1
    max_attempts = state.get("max_attempts", 3)

    node_span = telemetry.start_span("execute_task")

    result = None

    # Observe the real ToolNode result.
    for message in reversed(messages):
        if message.__class__.__name__ == "ToolMessage":
            result = message.content

            tool_name = getattr(message, "name", None) or "unknown_tool"
            tool_call_id = getattr(message, "tool_call_id", None)

            # Add the Qwen3 tool call to the observable trajectory.
            for previous_message in reversed(messages):
                tool_message_calls = getattr(
                    previous_message,
                    "tool_calls",
                    [],
                )

                if tool_message_calls:
                    tool_call = tool_message_calls[0]

                    tool_calls.append(
                        {
                            "tool": tool_call.get(
                                "name",
                                tool_name,
                            ),
                            "input": tool_call.get(
                                "args",
                                {},
                            ),
                            "trace_id": trace_id,
                            "span_id": tool_call_id,
                        }
                    )

                    break

            break

    # Determine whether any observed tool execution failed.
    failed_tool_results = [
        tool_result
        for tool_result in tool_results
        if tool_result.get("status") == "failed"
    ]

    tool_status = (
        "failed"
        if failed_tool_results
        else "completed"
    )

    tool_error = (
        failed_tool_results[-1].get("error")
        if failed_tool_results
        else None
    )

    # If no ToolMessage exists, the agent completed without a tool.
    last_message = messages[-1] if messages else None

    if last_message is not None:
        final_content = getattr(last_message, "content", None)

        if final_content:
            result = final_content

    trajectory.append(
        {
            "node": "execute_task",
            "status": tool_status,
            "task": task,
            "attempt": attempt_count,
            "trace_id": trace_id,
            "span_id": node_span["span_id"],
        }
    )

    node_event = telemetry.end_span(
        node_span,
        status=tool_status,
        error=tool_error,
        metadata={
            "node": "execute_task",
            "attempt": attempt_count,
        },
    )

    loop_analysis = detect_repeated_nodes(trajectory)

    reasoning_analysis = analyze_execution_reasoning(
        trajectory=trajectory,
        tool_calls=tool_calls,
        tool_results=tool_results,
        status=tool_status,
    )

    memory_analysis = analyze_memory(memory)

    trace_analysis = analyze_trace(
        trajectory=trajectory,
        tool_calls=tool_calls,
        tool_results=tool_results,
    )

    # Use actual measured tool latency when available.
    tool_latency_ms = None

    if tool_results:
        tool_latency_ms = tool_results[-1].get("latency_ms")

    failure_analysis = detect_failure_signals(
        tool_results=tool_results,
        status=tool_status,
        latency_ms=(
            tool_latency_ms
            if tool_latency_ms is not None
            else node_event["latency_ms"]
        ),
    )

    event = AgenticEvent(
        event_id=str(uuid.uuid4()),
        project_id="intellicore-agentic",
        model_id="qwen3:8b",
        trace_id=trace_id,
        span_id=node_event["span_id"],
        event_type="agent_execution",
        input={
            "task": task,
        },
        output=result,
        latency=node_event["latency_ms"],
        status=tool_status,
        error=tool_error,
        metadata={
            "node": "execute_task",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "trajectory": trajectory,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "loop_analysis": loop_analysis,
            "reasoning_analysis": reasoning_analysis,
            "memory_analysis": memory_analysis,
            "trace_analysis": trace_analysis,
            "failure_analysis": failure_analysis,
            "telemetry_events": telemetry.events,
        },
    )

    return {
        "current_node": "execute_task",
        "trajectory": trajectory,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "memory": memory,
        "memory_analysis": memory_analysis,
        "trace_analysis": trace_analysis,
        "output": result,
        "status": tool_status,
        "trace_id": trace_id,
        "span_id": node_event["span_id"],
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "loop_analysis": loop_analysis,
        "event": event.model_dump(mode="json"),
        "failure_analysis": failure_analysis,
        "messages": messages,
    }


def should_continue(state: AgentState) -> str:
    """Decide whether the agent should continue or finish."""

    if state.get("loop_analysis", {}).get("loop_detected"):
        return "end"

    if state.get("status") == "completed":
        return "end"

    if state.get("attempt_count", 0) >= state.get("max_attempts", 3):
        return "end"

    return "retry"


def build_agent_graph():
    """Build the IntelliCore Agentic execution graph."""

    graph = StateGraph(AgentState)

    graph.add_node("agent_llm", agent_llm)
    graph.add_node("tool_node", execute_tools)
    graph.add_node("execute_task", execute_task)

    graph.add_edge(START, "agent_llm")

    graph.add_conditional_edges(
        "agent_llm",
        route_after_llm,
        {
            "tool": "tool_node",
            "execute": "execute_task",
            "end": END,
        },
    )

    graph.add_edge("tool_node", "agent_llm")

    graph.add_edge("execute_task", END)

    return graph.compile()


agent_graph = build_agent_graph()
