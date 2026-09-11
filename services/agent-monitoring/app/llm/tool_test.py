from langchain_core.tools import tool

from .model import get_qwen3


@tool
def check_transaction_status(transaction_id: str) -> str:
    """Check the status of a transaction."""
    return f"Transaction {transaction_id} is completed."


model = get_qwen3()
model_with_tools = model.bind_tools([check_transaction_status])

response = model_with_tools.invoke(
    "Check the status of transaction TXN-001."
)

print("CONTENT:")
print(response.content)

print("\nTOOL CALLS:")
print(response.tool_calls)