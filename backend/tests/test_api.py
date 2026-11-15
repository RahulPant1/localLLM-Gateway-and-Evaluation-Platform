"""HTTP endpoints against the fake provider."""

from llm_gateway.providers.base import (
    LoadedModel,
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
