"""Guardrails added after a batch stalled WSL: memory guard, database lock, resume."""

import asyncio
import threading

from test_runs import make_batch, submit, wait_batch, wait_request

from llm_gateway.execution import ExecutionQueue, MemoryGuard
from llm_gateway.providers.base import (
    GenerationRequest,
    LoadedModel,
    LowMemory,
    ModelInfo,
    ProviderUnavailable,
)
from llm_gateway.schemas import PromptInput
from llm_gateway.store import Store
from llm_gateway.system import memory_mb

MODEL = ModelInfo(name="small:1b", size_bytes=1, num_ctx=4096)
LOADED = LoadedModel(name="small:1b", size_bytes=1, vram_bytes=1)


def guard(provider, readings, min_free_mb=1500):
    """A guard whose memory readings come from a list (the last one repeats)."""
    values = list(readings)
    read = lambda: values.pop(0) if len(values) > 1 else values[0]  # noqa: E731
    return MemoryGuard(provider, min_free_mb, read_available=read, settle_seconds=0.05)


def test_enough_memory_leaves_the_model_loaded(provider):
    provider.loaded = [LOADED]
    assert asyncio.run(guard(provider, [3000]).check()) == 3000
    assert provider.unloaded == []


def test_low_memory_unloads_the_model_and_carries_on(provider):
    provider.loaded = [LOADED]
    memory_guard = guard(provider, [900, 2800])
    assert asyncio.run(memory_guard.check()) == 2800
    assert provider.unloaded == ["small:1b"] and memory_guard.unloads == 1


def test_memory_still_critical_after_unloading_refuses_to_run(provider):
    provider.loaded = [LOADED]
    try:
        asyncio.run(guard(provider, [300]).check())
    except LowMemory as exc:
        assert exc.code == "low_memory" and "300 MB" in str(exc)
    else:
        raise AssertionError("expected LowMemory")
    assert provider.unloaded == ["small:1b"]


def test_low_but_not_critical_memory_with_nothing_loaded_still_runs(provider):
    assert asyncio.run(guard(provider, [900]).check()) == 900


def test_guard_can_be_switched_off_and_survives_an_unreachable_provider(provider):
    assert asyncio.run(guard(provider, [100], min_free_mb=0).check()) == 100
    provider.error = ProviderUnavailable("down")
    assert asyncio.run(guard(provider, [900]).check()) == 900


def test_queue_checks_memory_before_each_generation(provider):
    provider.loaded = [LOADED]
    queue = ExecutionQueue(provider, guard(provider, [900, 2500]))
    record = asyncio.run(queue.run(GenerationRequest(model="small:1b", prompt="hi"), MODEL))
    assert record.memory_available_mb == 2500
    assert queue.state().memory_unloads == 1


def test_low_memory_fails_the_run_with_a_clear_reason(make_client, provider, monkeypatch):
    monkeypatch.setattr("llm_gateway.execution.memory_available_mb", lambda: 200)
    client = make_client(min_free_memory_mb=1500)
    done = wait_request(client, submit(client)["id"])["executions"][0]
    assert done["status"] == "failed" and done["error_code"] == "low_memory"
    assert provider.calls == []  # the model was never started


def test_status_reports_memory_and_runs_record_it(make_client):
    client = make_client(min_free_memory_mb=1)
    memory = client.get("/api/status").json()["memory"]
    assert memory["guard_mb"] == 1
    assert memory["available_mb"] == memory_mb()[0] or memory["available_mb"] > 0
    done = wait_request(client, submit(client)["id"])["executions"][0]
    assert done["metrics"]["memory_available_mb"] > 0


def test_store_is_safe_when_read_from_many_threads(tmp_path):
    store = Store(tmp_path / "threads.db")
    for n in range(30):
        store.create_request(PromptInput(prompt=f"prompt {n}"), ["a", "b"])
    errors = []

    def read():
        try:
            for _ in range(40):
                assert len(store.list_requests(limit=100)) >= 30
        except Exception as exc:  # noqa: BLE001 - any failure is the bug
            errors.append(exc)

    def write():
        try:
            for n in range(40):
                store.create_request(PromptInput(prompt=f"more {n}"), ["a"])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(6)] + [threading.Thread(target=write)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(store.list_requests(limit=200)) == 70


def test_resume_reruns_only_what_was_interrupted(client, provider, tmp_path):
    batch = wait_batch(client, make_batch(client, prompts=["alpha", "beta"])["id"])
    assert batch["counts"]["completed"] == 4
    assert client.post(f"/api/batches/{batch['id']}/resume").json()["counts"]["completed"] == 4
    calls_before = len(provider.calls)

    # Simulate a crash that left two runs interrupted and one failed for a real reason.
    store = client.app.state.store
    ids = [e["id"] for r in batch["requests"] for e in r["executions"]]
    store.fail_execution(ids[0], "interrupted", "stopped")
    store.fail_execution(ids[1], "generation_timeout", "too slow")
    store.fail_execution(ids[2], "generation_failed", "model error")

    resumed = client.post(f"/api/batches/{batch['id']}/resume")
    assert resumed.status_code == 200
    final = wait_batch(client, batch["id"])
    assert final["counts"] == {"queued": 0, "running": 0, "completed": 3, "failed": 1}
    assert len(provider.calls) == calls_before + 2
    assert client.post("/api/batches/nope/resume").status_code == 404
