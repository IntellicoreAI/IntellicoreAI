import time
import uuid

from langgraph.graph import END, START, StateGraph

from ..loop_detector.detector import detect_repeated_nodes
from ..memory_analyzer.analyzer import analyze_memory
from ..reasoning_analyzer.analyzer import analyze_execution_reasoning
from ..signals.event_schema import AgenticEvent
from ..signals.failure_signals import detect_failure_signals
from ..tool_monitor.tools import check_transaction_status
from ..trace_analyzer.analyzer import analyze_trace
from .state import AgentState


def execute_task(state: AgentState) -> AgentState:
    """Execute the agent task and generate telemetry."""

    task = state.get("task", "")
    trajectory = state.get("trajectory", []).copy()
    tool_calls = state.get("tool_calls", []).copy()
    tool_results = state.get("tool_results", []).copy()
    memory = state.get("memory", []).copy()

    trace_id = state.get("trace_id") or str(uuid.uuid4())
    span_id = str(uuid.uuid4())

    attempt_count = state.get("attempt_count", 0) + 1
    max_attempts = state.get("max_attempts", 3)

    transaction_id = state.get("input_data", {}).get(
        "transaction_id",
        "TXN-001",
    )

    tool_calls.append(
        {
            "tool": "check_transaction_status",
            "input": {
                "transaction_id": transaction_id,
            },
            "trace_id": trace_id,
            "span_id": span_id,
        }
    )

    tool_start = time.perf_counter()

    try:
        result = check_transaction_status(transaction_id)
        tool_status = "completed"
        tool_error = None
    except Exception as exc:
        result = None
        tool_status = "failed"
        tool_error = str(exc)

    tool_latency_ms = (time.perf_counter() - tool_start) * 1000

    tool_results.append(
        {
            "tool": "check_transaction_status",
            "output": result,
            "status": tool_status,
            "latency_ms": tool_latency_ms,
            "error": tool_error,
            "trace_id": trace_id,
            "span_id": span_id,
        }
    )

    trajectory.append(
        {
            "node": "execute_task",
            "status": tool_status,
            "task": task,
            "tool": "check_transaction_status",
            "tool_status": tool_status,
            "tool_latency_ms": tool_latency_ms,
            "attempt": attempt_count,
            "trace_id": trace_id,
            "span_id": span_id,
        }
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

    failure_analysis = detect_failure_signals(
        tool_results=tool_results,
        status=tool_status,
        latency_ms=tool_latency_ms,
    )

    event = AgenticEvent(
        event_id=str(uuid.uuid4()),
        project_id="intellicore-agentic",
        model_id=None,
        trace_id=trace_id,
        span_id=span_id,
        event_type="tool_execution",
        input={
            "task": task,
            "transaction_id": transaction_id,
        },
        output=result,
        latency=tool_latency_ms,
        status=tool_status,
        error=tool_error,
        metadata={
            "node": "execute_task",
            "tool": "check_transaction_status",
            "tool_status": tool_status,
            "tool_latency_ms": tool_latency_ms,
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
        "span_id": span_id,
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "loop_analysis": loop_analysis,
        "event": event.model_dump(mode="json"),
        "failure_analysis": failure_analysis,
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

    graph.add_node("execute_task", execute_task)

    graph.add_edge(START, "execute_task")

    graph.add_conditional_edges(
        "execute_task",
        should_continue,
        {
            "retry": "execute_task",
            "end": END,
        },
    )

    return graph.compile()


agent_graph = build_agent_graph()