import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[2]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.memory_analyzer.analyzer import analyze_memory


def test_empty_memory():
    """Verify empty memory produces no available memory."""
    result = analyze_memory([])

    assert result["memory_available"] is False
    assert result["memory_count"] == 0
    assert result["memory_types"] == []
    assert result["memory_entries"] == []


def test_valid_memory_entries():
    """Verify valid memory entries are analyzed correctly."""
    memory = [
        {
            "type": "conversation",
            "content": "User prefers concise answers.",
        },
        {
            "type": "task_context",
            "content": "Transaction verification task.",
        },
    ]

    result = analyze_memory(memory)

    assert result["memory_available"] is True
    assert result["memory_count"] == 2
    assert result["memory_types"] == [
        "conversation",
        "task_context",
    ]
    assert result["memory_entries"] == memory


def test_invalid_memory_entries_are_ignored():
    """Verify non-dictionary memory entries are ignored."""
    memory = [
        {"type": "conversation", "content": "Valid entry"},
        "invalid-entry",
        None,
        123,
    ]

    result = analyze_memory(memory)

    assert result["memory_available"] is True
    assert result["memory_count"] == 1
    assert result["memory_types"] == ["conversation"]
    assert result["memory_entries"] == [
        {"type": "conversation", "content": "Valid entry"}
    ]