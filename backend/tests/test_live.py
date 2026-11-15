"""Smoke test against the real Ollama service. Loads a model onto the GPU.

Skipped by default. Run with:  uv run pytest -m live
"""

import asyncio

import httpx2
import pytest

from llm_gateway.catalog import ModelCatalog
from llm_gateway.config import get_settings
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import GenerationRequest
from llm_gateway.providers.ollama import OllamaProvider

pytestmark = pytest.mark.live
MODEL = "qwen3.5:4b"


def test_real_generation_end_to_end():
    async def scenario():
        settings = get_settings()
        async with httpx2.AsyncClient(base_url=settings.ollama_url) as client:
            provider = OllamaProvider(client, default_num_ctx=settings.default_num_ctx)
            model = await ModelCatalog(provider).get(MODEL)
            request = GenerationRequest(
                model=MODEL, prompt="What is the capital of France? Answer in one word."
            )
            record = await ExecutionQueue(provider).run(request, model)
            return record, await provider.list_loaded()

    record, loaded = asyncio.run(scenario())

    assert "paris" in record.result.response.lower()
    assert record.result.output_tokens > 0
    assert record.result.load_ms is not None
    assert record.execution_ms > 0
    assert MODEL in [m.name for m in loaded]
    # Reasoning must not leak into the answer.
    assert "<think>" not in record.result.response
