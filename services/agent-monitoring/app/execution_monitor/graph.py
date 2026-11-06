
import logging
import time
import uuid

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..analytics.analyzer import calculate_agent_metrics
from ..diagnosis.analyzer import diagnose_agent_execution
from ..instrumentation.telemetry import AgentTelemetry
from ..llm.model import get_qwen3
from ..loop_detector.detector import detect_repeated_nodes
from ..memory_analyzer.analyzer import analyze_memory
from ..reasoning_analyzer.analyzer import analyze_execution_reasoning
from ..recommendation.engine import generate_agent_recommendation
from ..signals.central_adapter import (
    normalize_agentic_event,
    publish_central_event,
)
from ..signals.event_publisher import publish_event
from ..signals.event_schema import AgenticEvent
from ..signals.failure_signals import detect_failure_signals
from ..tool_monitor.tools import check_transaction_status
from ..trace_analyzer.analyzer import analyze_trace
from .state import AgentState


logger = logging.getLogger(__name__)


def agent_llm(state: AgentState) -> AgentState:
    """Run the real local Qwen3 agent model."""

    execution_start_time = state.get("execution_start_time")

    if execution_start_time is None:
        execution_start_time = time.perf_counter()

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
        "execution_start_time": execution_start_time,
    }


def route_after_llm(state: AgentState) -> str:
    """Route the agent according to Qwen3's response."""

    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tool"

    return "execute"


def route_after_tools(state: AgentState) -> str:
    """Stop tool execution when a loop or the execution limit is reached."""

    trajectory = state.get("trajectory", [])
    tool_execution_count = state.get("tool_execution_count", 0)
    max_attempts = max(1, state.get("max_attempts", 3))

    loop_analysis = detect_repeated_nodes(trajectory)

    if loop_analysis.get("loop_detected"):
        return "execute"

    if tool_execution_count >= max_attempts:
        return "execute"

    return "agent_llm"


def execute_tools(state: AgentState) -> AgentState:
    """
    Execute the LangGraph ToolNode and capture IntelliCore telemetry.

    ToolNode may convert tool exceptions into ToolMessage error results.
    Those results are inspected without executing a tool twice.
    """

    trace_id = state.get("trace_id") or str(uuid.uuid4())
    telemetry = AgentTelemetry(trace_id=trace_id)
    tool_node = ToolNode([check_transaction_status])

    tool_span = telemetry.start_span("tool_node")

    try:
        result = tool_node.invoke(state)
        tool_status = "completed"
        tool_error = None

    except Exception as exc:
        result = {"messages": []}
        tool_status = "failed"
        tool_error = str(exc)

    messages = result.get("messages", [])

    # Inspect actual ToolMessage results.
    for message in messages:
        if (
            message.__class__.__name__ == "ToolMessage"
            and getattr(message, "status", None) == "error"
        ):
            tool_status = "failed"
            tool_error = str(getattr(message, "content", ""))
            break

    tool_event = telemetry.end_span(
        tool_span,
        status=tool_status,
        error=tool_error,
        metadata={"node": "tool_node"},
    )

    tool_name = "unknown_tool"
    tool_call_id = None
    tool_output = None

    # Identify the requested tool call.
    for message in reversed(state.get("messages", [])):
        tool_calls = getattr(message, "tool_calls", [])

        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name", "unknown_tool")
            tool_call_id = tool_call.get("id")
            break

    # Identify the corresponding tool result.
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

    telemetry_events = state.get("telemetry_events", []).copy()
    telemetry_events.append(tool_event)

    tool_execution_count = (
        state.get("tool_execution_count", 0) + 1
    )

    max_attempts = max(1, state.get("max_attempts", 3))
    loop_analysis = detect_repeated_nodes(trajectory)

    execution_stop_reason = state.get("execution_stop_reason")

    if loop_analysis.get("loop_detected"):
        execution_stop_reason = "Agentic loop detected."

    elif tool_execution_count >= max_attempts:
        execution_stop_reason = (
            "Maximum tool execution limit reached."
        )

    return {
        "messages": messages,
        "tool_results": tool_results,
        "trajectory": trajectory,
        "telemetry_events": telemetry_events,
        "trace_id": trace_id,
        "current_node": "tool_node",
        "execution_start_time": state.get("execution_start_time"),
        "tool_execution_count": tool_execution_count,
        "execution_stop_reason": execution_stop_reason,
    }


def execute_task(state: AgentState) -> AgentState:
    """
    Finalize execution and run IntelliCore analyses.

    This node records the result, telemetry, diagnosis,
    recommendations, analytics, and final Agentic event.
    """

    task = state.get("task", "")
    trajectory = state.get("trajectory", []).copy()
    tool_calls = state.get("tool_calls", []).copy()
    tool_results = state.get("tool_results", []).copy()
    telemetry_events = state.get("telemetry_events", []).copy()
    memory = state.get("memory", []).copy()
    messages = state.get("messages", [])

    trace_id = state.get("trace_id") or str(uuid.uuid4())
    telemetry = AgentTelemetry(trace_id=trace_id)

    attempt_count = state.get("attempt_count", 0) + 1
    max_attempts = max(1, state.get("max_attempts", 3))

    node_span = telemetry.start_span("execute_task")

    result = None

    # Record the latest ToolMessage and its corresponding tool call.
    for message in reversed(messages):
        if message.__class__.__name__ == "ToolMessage":
            result = message.content
            tool_name = getattr(message, "name", None) or "unknown_tool"
            tool_call_id = getattr(message, "tool_call_id", None)

            for previous_message in reversed(messages):
                message_tool_calls = getattr(
                    previous_message,
                    "tool_calls",
                    [],
                )

                matching_call = next(
                    (
                        call
                        for call in message_tool_calls
                        if call.get("id") == tool_call_id
                    ),
                    None,
                )

                if matching_call is not None:
                    tool_calls.append(
                        {
                            "tool": matching_call.get(
                                "name", tool_name
                            ),
                            "input": matching_call.get("args", {}),
                            "trace_id": trace_id,
                            "span_id": tool_call_id,
                        }
                    )
                    break

            break

    failed_tool_results = [
        tool_result
        for tool_result in tool_results
        if tool_result.get("status") == "failed"
    ]

    tool_status = (
        "failed" if failed_tool_results else "completed"
    )

    tool_error = (
        failed_tool_results[-1].get("error")
        if failed_tool_results
        else None
    )

    # A safety stop is a failed/incomplete execution, not a success.
    stop_reason = state.get("execution_stop_reason")

    if stop_reason:
        tool_status = "failed"
        tool_error = stop_reason

    # Prefer the final assistant response when available.
    if messages:
        last_message = messages[-1]

        if last_message.__class__.__name__ != "ToolMessage":
            final_content = getattr(last_message, "content", None)

            if final_content:
                result = final_content

        elif result is None:
            result = getattr(last_message, "content", None)

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

    telemetry_events.append(node_event)

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

    diagnosis = diagnose_agent_execution(
        failure_analysis=failure_analysis,
        tool_results=tool_results,
        trajectory=trajectory,
        reasoning_analysis=reasoning_analysis,
        loop_analysis=loop_analysis,
        trace_analysis=trace_analysis,
    )

    recommendation = generate_agent_recommendation(
        diagnosis=diagnosis,
        failure_analysis=failure_analysis,
    )

    execution_start_time = state.get("execution_start_time")

    if execution_start_time is None:
        execution_start_time = time.perf_counter()

    execution_latency_ms = (
        time.perf_counter() - execution_start_time
    ) * 1000

    analytics = calculate_agent_metrics(
        trajectory=trajectory,
        tool_calls=tool_calls,
        tool_results=tool_results,
        telemetry_events=telemetry_events,
        status=tool_status,
    )

    event = AgenticEvent(
        event_id=str(uuid.uuid4()),
        project_id="intellicore-agentic",
        model_id="qwen3:8b",
        trace_id=trace_id,
        span_id=node_event["span_id"],
        event_type="agent_execution",
        input={"task": task},
        output=result,
        latency=execution_latency_ms,
        status=tool_status,
        error=tool_error,
        metadata={
            "node": "execute_task",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "tool_execution_count": state.get(
                "tool_execution_count", 0
            ),
            "execution_stop_reason": stop_reason,
            "trajectory": trajectory,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "loop_analysis": loop_analysis,
            "reasoning_analysis": reasoning_analysis,
            "memory_analysis": memory_analysis,
            "trace_analysis": trace_analysis,
            "failure_analysis": failure_analysis,
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "telemetry_events": telemetry_events,
            "execution_latency_ms": execution_latency_ms,
            "analytics": analytics,
        },
    )

    # Preserve existing native Agentic AI event publishing.
    event_payload = event.model_dump(mode="json")
    publish_result = publish_event(event_payload)

    # Forward the monitoring result to central ingestion.
    try:
        monitoring_payload = normalize_agentic_event(
            event_payload,
            event_type="monitoring_result",
        )

        publish_central_event(monitoring_payload)

        logger.info(
            "Agentic monitoring event forwarded: %s",
            event.event_id,
        )

    except Exception:
        logger.exception(
            "Failed to forward Agentic monitoring event: %s",
            event.event_id,
        )

    # Forward an alert when existing failure detection identifies one.
    if failure_analysis.get("failure_detected", False):
        try:
            alert_payload = normalize_agentic_event(
                event_payload,
                event_type="alert",
            )

            publish_central_event(alert_payload)

            logger.warning(
                "Agentic alert forwarded: %s",
                alert_payload["event_id"],
            )

        except Exception:
            logger.exception(
                "Failed to forward Agentic alert: %s",
                event.event_id,
            )

    return {
        "current_node": "execute_task",
        "trajectory": trajectory,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "telemetry_events": telemetry_events,
        "memory": memory,
        "memory_analysis": memory_analysis,
        "trace_analysis": trace_analysis,
        "output": result,
        "status": tool_status,
        "trace_id": trace_id,
        "span_id": node_event["span_id"],
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "tool_execution_count": state.get(
            "tool_execution_count", 0
        ),
        "execution_stop_reason": stop_reason,
        "loop_analysis": loop_analysis,
        "event": event.model_dump(mode="json"),
        "failure_analysis": failure_analysis,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "analytics": analytics,
        "messages": messages,
        "execution_start_time": execution_start_time,
    }


def should_continue(state: AgentState) -> str:
    """Decide whether the agent should continue or finish."""

    if state.get("execution_stop_reason"):
        return "end"

    if state.get("loop_analysis", {}).get("loop_detected"):
        return "end"

    if state.get("status") == "completed":
        return "end"

    if state.get("attempt_count", 0) >= max(
        1, state.get("max_attempts", 3)
    ):
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

    # Bound the tool loop rather than returning to the LLM indefinitely.
    graph.add_conditional_edges(
        "tool_node",
        route_after_tools,
        {
            "agent_llm": "agent_llm",
            "execute": "execute_task",
        },
    )

    graph.add_edge("execute_task", END)

    return graph.compile()


agent_graph = build_agent_graph()
