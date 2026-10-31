from typing import Any

from langchain_core.tools import tool


@tool
def check_transaction_status(transaction_id: str) -> dict[str, Any]:
    """Check the status of a transaction."""

    if transaction_id == "INVALID-FAIL":
        raise RuntimeError(
            "Transaction service failed for test transaction INVALID-FAIL."
        )

    return {
        "transaction_id": transaction_id,
        "status": "completed",
    }