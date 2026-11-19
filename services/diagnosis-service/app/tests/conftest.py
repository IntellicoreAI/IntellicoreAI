import os
import pytest

@pytest.fixture(autouse=True)
def _isolate_storage(monkeypatch, request):
    """Unit tests run in-memory; tests marked `pg` use the DATABASE_URL given in the environment."""
    if "pg" not in request.keywords:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
