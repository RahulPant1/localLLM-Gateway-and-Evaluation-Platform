"""HTTP endpoints against the fake provider."""

import asyncio

import httpx2

from llm_gateway.main import create_app
from llm_gateway.providers.base import (
    GenerationFailed,
    GenerationTimeout,
    LoadedModel,
    ModelInfo,
    ProviderUnavailable,
)
from llm_gateway.system import parse_nvidia_smi


def test_models_lists_installed_and_marks_loaded(client, provider):
    provider.loaded = [LoadedModel(name="thinker:8b", size_bytes=5, vram_bytes=5)]
    body = client.get("/api/models").json()

    assert body["provider"] == "fake"
    assert {m["name"]: m["loaded"] for m in body["models"]} == {
        "small:1b": False,
        "thinker:8b": True,
    }


def test_models_are_cached_until_refresh(client, provider):
    client.get("/api/models")
    client.get("/api/models")
    assert provider.list_calls == 1
    client.get("/api/models", params={"refresh": True})
    assert provider.list_calls == 2


def test_models_when_provider_down(client, provider):
    provider.error = ProviderUnavailable("Cannot reach Ollama")
    response = client.get("/api/models")
    assert response.status_code == 503
    assert response.json()["error"] == "provider_unavailable"


def test_generate_returns_response_and_timings(client):
    response = client.post("/api/generate", json={"model": "small:1b", "prompt": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["response"] == "echo: hello"
    assert body["result"]["model"] == "small:1b"
    assert body["queue_wait_ms"] >= 0
    assert body["execution_ms"] >= 0
    assert body["started_at"]


def test_generate_unknown_model_is_404_without_running(client, provider):
    response = client.post("/api/generate", json={"model": "missing:1b", "prompt": "hello"})
    assert response.status_code == 404
    assert response.json()["error"] == "model_not_found"
    assert provider.calls == []


def test_generate_finds_a_newly_installed_model(client, provider):
    client.get("/api/models")  # fills the cache
    provider.models.append(ModelInfo(name="new:1b", size_bytes=1, num_ctx=4096))
    response = client.post("/api/generate", json={"model": "new:1b", "prompt": "hello"})
    assert response.status_code == 200


def test_generate_rejects_empty_prompt(client):
    response = client.post("/api/generate", json={"model": "small:1b", "prompt": ""})
    assert response.status_code == 422


def test_generate_error_statuses(client, provider):
    client.get("/api/models")  # model lookup succeeds; only generation fails
    for error, status, code in [
        (GenerationFailed("oom"), 502, "generation_failed"),
        (GenerationTimeout("slow"), 504, "generation_timeout"),
        (ProviderUnavailable("down"), 503, "provider_unavailable"),
    ]:
        provider.error = error
        provider.list_models = _static(provider.models)
        response = client.post("/api/generate", json={"model": "small:1b", "prompt": "hi"})
        assert (response.status_code, response.json()["error"]) == (status, code)


def _static(models):
    async def list_models():
        return list(models)

    return list_models


def test_a_failed_generation_does_not_block_the_next(client, provider):
    client.get("/api/models")
    provider.list_models = _static(provider.models)
    provider.error = GenerationFailed("oom")
    assert (
        client.post("/api/generate", json={"model": "small:1b", "prompt": "a"}).status_code == 502
    )
    provider.error = None
    assert (
        client.post("/api/generate", json={"model": "small:1b", "prompt": "b"}).status_code == 200
    )
    assert client.get("/api/status").json()["queue"] == {
        "busy": False,
        "running_model": None,
        "waiting": 0,
    }


def test_generations_run_one_at_a_time_in_order(provider):
    provider.delay = 0.02
    app = create_app(provider=provider)

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(transport=transport, base_url="http://test") as http:
                names = ["small:1b", "thinker:8b", "small:1b"]
                tasks = []
                for name in names:
                    tasks.append(
                        asyncio.create_task(
                            http.post("/api/generate", json={"model": name, "prompt": "hi"})
                        )
                    )
                    await asyncio.sleep(0.005)  # fix the arrival order
                return [r.json() for r in await asyncio.gather(*tasks)]

    records = asyncio.run(scenario())

    assert provider.max_running == 1
    assert provider.calls == ["small:1b", "thinker:8b", "small:1b"]
    waits = [r["queue_wait_ms"] for r in records]
    assert waits[0] < waits[1] < waits[2]
    assert waits[2] >= 20  # waited for at least one full generation


def test_status_ok(client, provider):
    provider.loaded = [LoadedModel(name="small:1b", size_bytes=10, vram_bytes=10)]
    body = client.get("/api/status").json()

    assert body["status"] == "ok"
    assert body["provider"]["reachable"] is True
    assert body["provider"]["version"] == "0.0-fake"
    assert body["provider"]["loaded_models"][0]["name"] == "small:1b"
    assert body["queue"]["busy"] is False
    assert body["evaluator"] == {"configured": False, "judge_model": "claude-opus-5-5"}
    assert "available" in body["gpu"]


def test_status_degrades_when_provider_down(client, provider):
    provider.error = ProviderUnavailable("Cannot reach Ollama: refused")
    response = client.get("/api/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["provider"]["reachable"] is False
    assert "refused" in body["provider"]["error"]


def test_status_never_exposes_api_key(make_client):
    secret = "sk-ant-test-not-a-real-key"
    response = make_client(anthropic_api_key=secret).get("/api/status")
    assert response.json()["evaluator"]["configured"] is True
    assert secret not in response.text


def test_parse_nvidia_smi():
    gpu = parse_nvidia_smi("NVIDIA GeForce RTX 4060 Laptop GPU, 420, 8188, 8\n")
    assert gpu.available is True
    assert (gpu.memory_used_mib, gpu.memory_total_mib, gpu.utilization_percent) == (420, 8188, 8)

    for bad in ("", "garbage", "GPU, [N/A], 8188, 8"):
        assert parse_nvidia_smi(bad).available is False
