"""OllamaProvider against a fake Ollama HTTP server (no network, no GPU)."""

import asyncio
import json

import httpx2
import pytest

from llm_gateway.providers.base import (
    GenerationFailed,
    GenerationRequest,
    GenerationTimeout,
    ModelInfo,
    ModelNotFound,
    ProviderUnavailable,
)
from llm_gateway.providers.ollama import OllamaProvider

TAGS = {
    "models": [
        {
            "name": "qwen3.5:4b",
            "size": 3_300_000_000,
            "details": {"family": "qwen35", "parameter_size": "4.2B", "quantization_level": "Q4"},
        },
        {"name": "med:4b", "size": 4_100_000_000, "details": {"family": "gemma3"}},
        {"name": "broken:1b", "size": 1, "details": {}},
    ]
}
SHOW = {
    "qwen3.5:4b": {
        "capabilities": ["completion", "thinking"],
        "parameters": "top_k   20\ntemperature   1",
        "model_info": {"qwen35.context_length": 262144},
    },
    "med:4b": {
        "capabilities": ["completion"],
        "parameters": 'num_ctx                        4096\nstop   "<end_of_turn>"',
        "model_info": {"gemma3.context_length": 131072},
    },
}
CHAT_OK = {
    "message": {"role": "assistant", "content": "Paris.", "thinking": "Capital of France..."},
    "done_reason": "stop",
    "total_duration": 5_000_000_000,
    "load_duration": 3_000_000_000,
    "prompt_eval_count": 12,
    "prompt_eval_duration": 100_000_000,
    "eval_count": 40,
    "eval_duration": 2_000_000_000,
}


def make_provider(chat=None, **kwargs):
    """Return (provider, list of chat payloads the fake server received)."""
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx2.Response(200, json={"version": "0.40.3"})
        if path == "/api/tags":
            return httpx2.Response(200, json=TAGS)
        if path == "/api/ps":
            return httpx2.Response(
                200,
                json={
                    "models": [
                        {"name": "med:4b", "size": 5, "size_vram": 4, "context_length": 4096}
                    ]
                },
            )
        if path == "/api/show":
            name = json.loads(request.content)["model"]
            if name in SHOW:
                return httpx2.Response(200, json=SHOW[name])
            return httpx2.Response(500, json={"error": "boom"})
        if path == "/api/chat":
            seen.append(json.loads(request.content))
            return chat(request) if chat else httpx2.Response(200, json=CHAT_OK)
        return httpx2.Response(404)

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), base_url="http://ollama.test"
    )
    return OllamaProvider(client, **kwargs), seen


THINKER = ModelInfo(name="qwen3.5:4b", size_bytes=1, capabilities=["thinking"], num_ctx=8192)
PLAIN = ModelInfo(name="med:4b", size_bytes=1, capabilities=["completion"], num_ctx=4096)


def run(coro):
    return asyncio.run(coro)


def test_discovery_reads_details_and_context_sizes():
    provider, _ = make_provider()
    models = {m.name: m for m in run(provider.list_models())}

    assert set(models) == {"qwen3.5:4b", "med:4b", "broken:1b"}
    qwen = models["qwen3.5:4b"]
    assert qwen.capabilities == ["completion", "thinking"]
    assert qwen.max_context_length == 262144
    assert qwen.num_ctx == 8192  # gateway default, below the model's limit
    assert models["med:4b"].num_ctx == 4096  # the Modelfile value wins


def test_model_whose_details_fail_is_still_listed():
    provider, _ = make_provider()
    broken = next(m for m in run(provider.list_models()) if m.name == "broken:1b")
    assert broken.details_available is False
    assert broken.capabilities == []


def test_default_num_ctx_is_capped_by_model_limit():
    provider, _ = make_provider(default_num_ctx=1_000_000)
    qwen = next(m for m in run(provider.list_models()) if m.name == "qwen3.5:4b")
    assert qwen.num_ctx == 262144


def test_version_and_loaded_models():
    provider, _ = make_provider()
    assert run(provider.version()) == "0.40.3"
    loaded = run(provider.list_loaded())
    assert [(m.name, m.vram_bytes) for m in loaded] == [("med:4b", 4)]


def test_generate_separates_thinking_and_converts_timings():
    provider, seen = make_provider()
    result = run(
        provider.generate(GenerationRequest(model="qwen3.5:4b", prompt="Capital?"), THINKER)
    )

    assert result.response == "Paris."
    assert result.thinking == "Capital of France..."
    assert result.load_ms == 3000.0
    assert result.generation_ms == 2000.0
    assert result.provider_total_ms == 5000.0
    assert result.tokens_per_second == 20.0
    assert result.warnings == []
    assert seen[0]["think"] is True
    assert seen[0]["stream"] is False
    assert seen[0]["options"] == {"num_ctx": 8192}
    assert "keep_alive" not in seen[0]


def test_generate_does_not_send_think_to_models_without_it():
    provider, seen = make_provider()
    request = GenerationRequest(model="med:4b", prompt="Hi", system="Be brief", max_tokens=5)
    run(provider.generate(request, PLAIN))

    assert "think" not in seen[0]
    assert seen[0]["messages"][0] == {"role": "system", "content": "Be brief"}
    assert seen[0]["options"] == {"num_ctx": 4096, "num_predict": 5}


def test_think_can_be_switched_off():
    provider, seen = make_provider()
    run(provider.generate(GenerationRequest(model="qwen3.5:4b", prompt="Hi", think=False), THINKER))
    assert seen[0]["think"] is False


def test_truncation_warnings():
    def full(_):
        body = CHAT_OK | {"done_reason": "length", "prompt_eval_count": 4000, "eval_count": 96}
        return httpx2.Response(200, json=body)

    provider, _ = make_provider(chat=full)
    request = GenerationRequest(model="med:4b", prompt="x" * 20_000)
    result = run(provider.generate(request, PLAIN))
    assert set(result.warnings) == {
        "prompt_probably_truncated",
        "output_cut_off",
        "context_window_full",
    }


def raises(exc_type):
    def chat(request):
        raise exc_type("simulated", request=request)

    return chat


@pytest.mark.parametrize(
    ("chat", "expected"),
    [
        (lambda _: httpx2.Response(404, json={"error": "model 'x' not found"}), ModelNotFound),
        (lambda _: httpx2.Response(500, json={"error": "out of memory"}), GenerationFailed),
        (lambda _: httpx2.Response(200, text="not json"), GenerationFailed),
        (raises(httpx2.ConnectError), ProviderUnavailable),
        (raises(httpx2.ReadTimeout), GenerationTimeout),
        (raises(httpx2.ReadError), GenerationFailed),
    ],
)
def test_generate_error_mapping(chat, expected):
    provider, _ = make_provider(chat=chat)
    with pytest.raises(expected) as info:
        run(provider.generate(GenerationRequest(model="med:4b", prompt="Hi"), PLAIN))
    assert type(info.value) is expected


def test_unreachable_ollama_raises_provider_unavailable():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(refuse), base_url="http://x.test")
    provider = OllamaProvider(client)
    for call in (provider.version, provider.list_models, provider.list_loaded):
        with pytest.raises(ProviderUnavailable):
            run(call())
