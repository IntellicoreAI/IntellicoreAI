from typing import Any


def analyze_memory(
    memory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Analyze observable memory entries from an Agentic execution.

    This analyzes structured memory data supplied by the agent.
    It does not inspect private model memory or hidden state.
    """

    memory = memory or []

    valid_entries = [
        entry for entry in memory
        if isinstance(entry, dict)
    ]

    memory_types = [
        entry.get("type")
        for entry in valid_entries
        if entry.get("type")
    ]

    return {
        "memory_available": bool(valid_entries),
        "memory_count": len(valid_entries),
        "memory_types": memory_types,
        "memory_entries": valid_entries,
    }