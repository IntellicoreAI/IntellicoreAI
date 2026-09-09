from collections import Counter
from typing import Any


def detect_repeated_nodes(
    trajectory: list[dict[str, Any]],
    threshold: int = 3,
) -> dict[str, Any]:
    """
    Detect repeated node executions in an agent trajectory.

    A node is considered to be looping when it appears at least
    `threshold` times in the trajectory.
    """

    if not trajectory:
        return {
            "loop_detected": False,
            "repeated_nodes": [],
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

    return {
        "loop_detected": bool(repeated_nodes),
        "repeated_nodes": repeated_nodes,
        "message": (
            "Repeated node execution detected."
            if repeated_nodes
            else "No repeated node execution detected."
        ),
    }