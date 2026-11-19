import os
import pytest

# Never let tests try to reach a real central ingestion service.
os.environ.setdefault("CENTRAL_INGESTION_ENABLED", "false")


@pytest.fixture(scope="session", autouse=True)
def _init_llm_storage():
    """TestClient(app) without a `with` block skips the app lifespan, so create the
    LLM-owned tables (llm_alerts / llm_feedback) explicitly on a fresh database."""
    from app.storage import init_storage
    import app.main  # noqa: F401  (loads .env so DATABASE_URL is honoured)
    init_storage()
