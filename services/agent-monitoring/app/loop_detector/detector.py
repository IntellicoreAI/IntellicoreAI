from collections import Counter
from typing import Any


def detect_repeated_nodes(
    trajectory: list[dict[str, Any]],
    threshold: int = 3,
) -> dict[str, Any]:
    """
    Detect Agentic loops.

    Supports:

    1. Repeated nodes:
       tool → tool → tool

    2. Alternating cycles:
       llm → tool → llm → tool
    """

    if not trajectory:
        return {
            "loop_detected": False,
            "loop_type": "none",
            "repeated_nodes": [],
            "cycle_pattern": [],
            "message": "No trajectory data available.",
        }

    node_names = [
        step.get("node")
        for step in trajectory
        if step.get("node")
    ]

    counts = Counter(node_names)

    repeated_nodes = [
        {
            "node": node,
            "count": count,
        }
        for node, count in counts.items()
        if count >= threshold
    ]

    cycle_pattern = []

    if len(node_names) >= 4:

        pattern = node_names[-4:]

        if (
            pattern[0] == pattern[2]
            and pattern[1] == pattern[3]
            and pattern[0] != pattern[1]
        ):
            cycle_pattern = pattern

    loop_detected = (
        bool(repeated_nodes)
        or bool(cycle_pattern)
    )

    loop_type = "none"

    if repeated_nodes:
        loop_type = "repeated_node"

    elif cycle_pattern:
        loop_type = "alternating_cycle"

    return {
        "loop_detected": loop_detected,
        "loop_type": loop_type,
        "repeated_nodes": repeated_nodes,
        "cycle_pattern": cycle_pattern,
        "message": (
            "Agentic loop detected."
            if loop_detected
            else "No loop detected."
        ),
    }