import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from llm_gateway.config import Settings, get_settings
from llm_gateway.main import create_app
from llm_gateway.providers.base import (
    GenerationRequest,
    GenerationResult,
    LoadedModel,
    ModelInfo,
    ModelProvider,
)


class FakeProvider(ModelProvider):
    """In-memory provider. Set `error` to make every call fail."""

    name = "fake"

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self.models = (
            models
            if models is not None
            else [
                ModelInfo(
                    name="small:1b", size_bytes=1_000, capabilities=["completion"], num_ctx=4096
                ),
                ModelInfo(
                    name="thinker:8b",
                    size_bytes=5_000,
                    capabilities=["completion", "thinking"],
                    num_ctx=8192,
                ),
            ]
        )
        self.loaded: list[LoadedModel] = []
        self.error: Exception | None = None
        self.delay = 0.0
        self.running = 0
        self.max_running = 0
        self.calls: list[str] = []
        self.list_calls = 0

    async def version(self) -> str:
        if self.error:
            raise self.error
        return "0.0-fake"

    async def list_models(self) -> list[ModelInfo]:
        self.list_calls += 1
        if self.error:
            raise self.error
        return list(self.models)

    async def list_loaded(self) -> list[LoadedModel]:
        if self.error:
            raise self.error
        return list(self.loaded)

    async def generate(self, request: GenerationRequest, model: ModelInfo) -> GenerationResult:
        if self.error:
            raise self.error
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.running -= 1
        self.calls.append(model.name)
        return GenerationResult(
            model=model.name, response=f"echo: {request.prompt}", num_ctx=model.num_ctx
        )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def make_client(tmp_path, provider, request):
    """Build a test client with explicit settings and the fake provider.

    `_env_file=None` stops the developer's real `.env` (and any real API key)
    from leaking into tests.
    """

    def _make(**overrides) -> TestClient:
        overrides.setdefault("database_path", tmp_path / "test.db")
        settings = Settings(_env_file=None, **overrides)
        app = create_app(provider=provider)
        app.dependency_overrides[get_settings] = lambda: settings
        client = TestClient(app)
        client.__enter__()  # runs the app's startup
        request.addfinalizer(lambda: client.__exit__(None, None, None))
        return client

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture(autouse=True)
def _no_gateway_env(monkeypatch):
    """Real GATEWAY_* environment variables must not affect tests."""
    for name in list(os.environ):
        if name.startswith("GATEWAY_"):
            monkeypatch.delenv(name)
