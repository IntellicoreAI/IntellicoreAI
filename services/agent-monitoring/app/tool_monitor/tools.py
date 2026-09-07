from typing import Any


def check_transaction_status(transaction_id: str) -> dict[str, Any]:
    """Check the status of a transaction."""

    return {
        "transaction_id": transaction_id,
        "status": "completed",
    }