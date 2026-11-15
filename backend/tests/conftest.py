import asyncio
import os

import pytest
from fastapi.testclient import TestClient

from llm_gateway.config import Settings
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
        self.unloaded: list[str] = []
        self.requests: list[GenerationRequest] = []
        # What the fake returns when asked to classify a request (schema-constrained call).
        self.classifier_answer = '{"category": "general_qa", "complexity": "low"}'
        self.classifier_calls = 0
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

    async def unload(self, model: str) -> None:
        self.unloaded.append(model)
        self.loaded = [m for m in self.loaded if m.name != model]

    async def list_loaded(self) -> list[LoadedModel]:
        if self.error:
            raise self.error
        return list(self.loaded)

    async def generate(
        self, request: GenerationRequest, model: ModelInfo, on_delta=None
    ) -> GenerationResult:
        if self.error:
            raise self.error
        if request.response_schema:
            self.classifier_calls += 1
            return GenerationResult(
                model=model.name, response=self.classifier_answer, num_ctx=model.num_ctx
            )
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.running -= 1
        self.calls.append(model.name)
        self.requests.append(request)
        answer = f"echo: {request.prompt}"
        if on_delta:
            if request.think:
                on_delta("thinking", "hmm")
            on_delta("content", answer[:4])
            on_delta("content", answer[4:])
        return GenerationResult(
            model=model.name,
            response=answer,
            num_ctx=model.num_ctx,
            prompt_tokens=7,
            output_tokens=3,
            done_reason="stop",
        )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


TEST_REGISTRY = """
classifier_model: small:1b
models:
  small:1b: {strengths: Quick general model}
  thinker:8b: {strengths: Careful reasoning}
policy:
  general_qa: {default: [small:1b, thinker:8b]}
  classification: {default: [small:1b]}
  extraction: {default: [small:1b]}
  structured_output: {default: [[small:1b, thinker:8b]]}
  reasoning: {default: [thinker:8b, small:1b]}
  coding: {default: [coder:7b, thinker:8b, small:1b]}
  medical: {default: [medic:4b]}
last_resort: [small:1b]
"""


@pytest.fixture
def registry_path(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(TEST_REGISTRY)
    return path


@pytest.fixture
def make_client(tmp_path, provider, request, registry_path):
    """Build a test client with explicit settings and the fake provider.

    `_env_file=None` stops the developer's real `.env` (and any real API key)
    from leaking into tests.
    """

    def _make(**overrides) -> TestClient:
        overrides.setdefault("database_path", tmp_path / "test.db")
        overrides.setdefault("registry_path", registry_path)
        # The memory guard reads the real machine; tests that need it enable it.
        overrides.setdefault("min_free_memory_mb", 0)
        settings = Settings(_env_file=None, **overrides)
        app = create_app(provider=provider, settings=settings)
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
