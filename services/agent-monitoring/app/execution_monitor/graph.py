import time
import uuid

from langgraph.graph import END, START, StateGraph

from ..signals.event_schema import AgenticEvent
from ..tool_monitor.tools import check_transaction_status
from .state import AgentState


def execute_task(state: AgentState) -> AgentState:
    """Execute the agent task and generate telemetry."""

    task = state.get("task", "")
    trajectory = state.get("trajectory", []).copy()
    tool_calls = state.get("tool_calls", []).copy()
    tool_results = state.get("tool_results", []).copy()

    trace_id = state.get("trace_id") or str(uuid.uuid4())
    span_id = str(uuid.uuid4())

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
            "trace_id": trace_id,
            "span_id": span_id,
        }
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
            "trajectory": trajectory,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        },
    )

    return {
        "current_node": "execute_task",
        "trajectory": trajectory,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "output": result,
        "status": tool_status,
        "trace_id": trace_id,
        "span_id": span_id,
        "event": event.model_dump(),
    }


def build_agent_graph():
    """Build the IntelliCore Agentic execution graph."""

    graph = StateGraph(AgentState)

    graph.add_node("execute_task", execute_task)

    graph.add_edge(START, "execute_task")
    graph.add_edge("execute_task", END)

    return graph.compile()


agent_graph = build_agent_graph()