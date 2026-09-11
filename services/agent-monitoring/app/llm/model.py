from langchain_ollama import ChatOllama


QWEN3_MODEL = "qwen3:8b"


def get_qwen3():
    """Return the local Qwen3 model used by the Agentic runtime."""

    return ChatOllama(
        model=QWEN3_MODEL,
        temperature=0,
        disable_streaming=True,
    )