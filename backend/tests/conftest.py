import pytest
from fastapi.testclient import TestClient

from llm_gateway.config import Settings, get_settings
from llm_gateway.main import create_app


@pytest.fixture
def make_client(tmp_path):
    """Build a test client with explicit settings.

    `_env_file=None` stops the developer's real `.env` (and any real API key)
    from leaking into tests.
    """

    def _make(**overrides) -> TestClient:
        overrides.setdefault("database_path", tmp_path / "test.db")
        settings = Settings(_env_file=None, **overrides)
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)

    return _make


@pytest.fixture(autouse=True)
def _no_gateway_env(monkeypatch):
    """Real GATEWAY_* environment variables must not affect tests."""
    import os

    for name in list(os.environ):
        if name.startswith("GATEWAY_"):
            monkeypatch.delenv(name)
